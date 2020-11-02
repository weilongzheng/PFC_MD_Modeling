# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Model of PFC-MD with learning on output weights as in Rikhye, Gilra and Halassa, Nature Neuroscience 2018."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import sys,shelve
import plot_utils as pltu
import scipy.io as sio
import pickle

class PFCMD():
    def __init__(self,PFC_G,PFC_G_off,learning_rate,
                    noiseSD,tauError,plotFigs=True,saveData=False):
        self.RNGSEED = 9
        np.random.seed([self.RNGSEED])

        self.Nsub = 200                     # number of neurons per cue
        self.Ntasks = 2                     # number of contexts = number of MD cells.
        self.xorTask = False                # use xor Task or simple 1:1 map task
        #self.xorTask = True                 # use xor Task or simple 1:1 map task
        self.Ncues = self.Ntasks*2          # number of input cues
        self.Nneur = self.Nsub*(self.Ncues+1)# number of neurons
        if self.xorTask: self.inpsPerTask = 4# number of cue combinations per task
        else: self.inpsPerTask = 2
        self.Nout = 2                       # number of outputs
        self.tau = 0.02
        self.dt = 0.001
        self.tsteps = 200                   # number of timesteps in a trial
        self.cuesteps = 100                 # number of time steps for which cue is on
        self.noiseSD = noiseSD
        self.saveData = saveData

        self.learning_rate = learning_rate  # too high a learning rate makes the output weights
                                            #  change too much within a trial / training cycle,
                                            #  then the output interference depends
                                            #  on the order of cues within a cycle
                                            # typical values is 1e-5, can vary from 1e-4 to 1e-6
        self.tauError = tauError            # smooth the error a bit, so that weights don't fluctuate

        self.positiveRates = True           # whether to clip rates to be only positive, G must also change
        
        #self.MDstrength = None              # if None, use wPFC2MD, if not None as below, just use context directly
        #self.MDstrength = 0.                # a parameter that controls how much the MD disjoints task representations.
        self.MDstrength = 1.                # a parameter that controls how much the MD disjoints task representations.
                                            #  zero would be a pure reservoir, 1 would be full MDeffect
                                            # -1 for zero recurrent weights
        self.blockTrain = True              # first half of training is context1, second half is context2
        
        self.wPFC2MD = np.zeros(shape=(self.Ntasks,self.Nneur))
        for taski in np.arange(self.Ntasks):
            self.wPFC2MD[taski,self.Nsub*taski*2:self.Nsub*(taski+1)*2] = 1./self.Nsub

        # multiply rec wts for current context neurons, subtractive inh for other neurons
        Gbase = 0.75                      # determines also the cross-task recurrence
        if self.MDstrength is None: MDval = 1.
        elif self.MDstrength < 0.: MDval = 0.
        else: MDval = self.MDstrength
        # subtract across tasks (task with higher MD suppresses cross-tasks)
        self.wMD2PFC = np.ones(shape=(self.Nneur,self.Ntasks)) * (-10.) * MDval
        for taski in np.arange(self.Ntasks):
            self.wMD2PFC[self.Nsub*2*taski:self.Nsub*2*(taski+1),taski] = 0.
        self.useMult = True
        # multiply recurrence within task, no addition across tasks
        self.wMD2PFCMult = np.ones(shape=(self.Nneur,self.Ntasks)) \
                            * PFC_G_off/Gbase * (1-MDval)
        for taski in np.arange(self.Ntasks):
            self.wMD2PFCMult[self.Nsub*2*taski:self.Nsub*2*(taski+1),taski]\
                        += PFC_G/Gbase * MDval

        # Choose G based on the type of activation function
        # unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if self.positiveRates:
            self.G = Gbase
            self.tauMD = self.tau
        else:
            self.G = Gbase
            self.tauMD = self.tau*10

        # Perhaps I shouldn't have self connections / autapses?!
        # Perhaps I should have sparse connectivity?
        self.Jrec = np.random.normal(size=(self.Nneur, self.Nneur))\
                        *self.G/np.sqrt(self.Nsub*2)
        if self.MDstrength < 0.: self.Jrec *= 0.

        # make mean input to each row zero,
        #  helps to avoid saturation (both sides) for positive-only rates.
        #  see Nicola & Clopath 2016
        # mean of rows i.e. across columns (axis 1),
        #  then expand with np.newaxis
        #   so that numpy's broadcast works on rows not columns
        self.Jrec -= np.mean(self.Jrec,axis=1)[:,np.newaxis]

        # I don't want to have an if inside activation
        #  as it is called at each time step of the simulation
        # But just defining within __init__
        #  doesn't make it a member method of the class,
        #  hence the special self.__class__. assignment
        if self.positiveRates:
            # only +ve rates
            def activation(self,inp):
                return np.clip(np.tanh(inp),0,None)
                #return np.sqrt(np.clip(inp,0,None))
                #return (np.tanh(inp)+1.)/2.
        else:
            # both +ve/-ve rates as in Miconi
            def activation(self,inp):
                return np.tanh(inp)
        self.__class__.activation = activation

        self.wIn = np.zeros((self.Nneur,self.Ncues))
        self.cueFactor = 1.5
        if self.positiveRates: lowcue,highcue = 0.5,1.
        else: lowcue,highcue = -1.,1
        for cuei in np.arange(self.Ncues):
            self.wIn[self.Nsub*cuei:self.Nsub*(cuei+1),cuei] = \
                    np.random.uniform(lowcue,highcue,size=self.Nsub) \
                            *self.cueFactor

        self.cue_eigvecs = np.zeros((self.Ncues,self.Nneur))
        self.plotFigs = plotFigs
        self.cuePlot = (0,0)
                
        if self.saveData:
            self.fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD'+\
                                    str(self.MDstrength)+\
                                    '_R'+str(self.RNGSEED)+\
                                    ('_xor' if self.xorTask else '')+'.shelve')

        self.meanAct = np.zeros(shape=(self.Ntasks*self.inpsPerTask,\
                                    self.tsteps,self.Nneur))

    def sim_cue(self,taski,cuei,cue,target,
                    train=True,routsTarget=None):
        cues = np.zeros(shape=(self.tsteps,self.Ncues))
        # random initialization of input to units
        # very important to have some random input
        #  just for the xor task for (0,0) cue!
        #  keeping it also for the 1:1 task just for consistency
        xinp = np.random.uniform(0,0.1,size=(self.Nneur))
        #xinp = np.zeros(shape=(self.Nneur))
        xadd = np.zeros(shape=(self.Nneur))
        routs = np.zeros(shape=(self.tsteps,self.Nneur))
        MDouts = np.zeros(shape=(self.tsteps,self.Ntasks))
        outInp = np.zeros(shape=self.Nout)
        outs = np.zeros(shape=(self.tsteps,self.Nout))
        out = np.zeros(self.Nout)
        errors = np.zeros(shape=(self.tsteps,self.Nout))
        error_smooth = np.zeros(shape=self.Nout)

        for i in range(self.tsteps):
            #import pdb;pdb.set_trace()
            rout = self.activation(xinp)
            routs[i,:] = rout
            outAdd = np.dot(self.wOut,rout)

            MDout = np.zeros(self.Ntasks)
            MDout[taski] = 1.
            MDouts[i,:] = MDout

            if self.useMult:
                self.MD2PFCMult = np.dot(self.wMD2PFCMult,MDout)
                xadd = (1.+self.MD2PFCMult) * np.dot(self.Jrec,rout)
            else:
                xadd = np.dot(self.Jrec,rout)
            xadd += np.dot(self.wMD2PFC,MDout)

            if i < self.cuesteps:
                # baseline cue is always added
                xadd += np.dot(self.wIn,cue)
                cues[i,:] = cue

            xinp += self.dt/self.tau * (-xinp + xadd)
        
            outInp += self.dt/self.tau * (-outInp + outAdd)
            out = self.activation(outInp)                
            error = out - target
            errors[i,:] = error
            outs[i,:] = out
            error_smooth += self.dt/self.tauError * (-error_smooth + error)
        
            if train:
                # error-driven i.e. error*pre (perceptron like) learning
                self.wOut += -self.learning_rate \
                                * np.outer(error_smooth,rout)

            inpi = taski*self.inpsPerTask + cuei

        self.meanAct[inpi,:,:] += routs

        return cues, routs, outs, MDouts, errors

    def get_cues_order(self,cues):
        cues_order = np.random.permutation(cues)
        return cues_order

    def get_cue_target(self,taski,cuei):
        cue = np.zeros(self.Ncues)
        inpBase = taski*2
        if cuei in (0,1):
            cue[inpBase+cuei] = 1.
        elif cuei == 3:
            cue[inpBase:inpBase+2] = 1
        
        if self.xorTask:
            if cuei in (0,1):
                target = np.array((1.,0.))
            else:
                target = np.array((0.,1.))
        else:
            if cuei == 0: target = np.array((1.,0.))
            else: target = np.array((0.,1.))
        return cue, target

    def plot_column(self,fig,cues,routs,MDouts,outs,ploti=0):
        print('Plotting ...')
        cols=4
        if ploti==0:
            yticks = (0,1)
            ylabels=('Cues','PFC for cueA','PFC for cueB',
                        'PFC for cueC','PFC for cueD','PFC for rest',
                        'MD 1,2','Output 1,2')
        else:
            yticks = ()
            ylabels=('','','','','','','','')
        ax = fig.add_subplot(8,cols,1+ploti)
        ax.plot(cues,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[0])
        ax = fig.add_subplot(8,cols,cols+1+ploti)
        ax.plot(routs[:,:10],linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[1])
        ax = fig.add_subplot(8,cols,cols*2+1+ploti)
        ax.plot(routs[:,self.Nsub:self.Nsub+10],
                    linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[2])
        if self.Ncues > 2:
            ax = fig.add_subplot(8,cols,cols*3+1+ploti)
            ax.plot(routs[:,self.Nsub*2:self.Nsub*2+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[3])
            ax = fig.add_subplot(8,cols,cols*4+1+ploti)
            ax.plot(routs[:,self.Nsub*3:self.Nsub*3+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[4])
            ax = fig.add_subplot(8,cols,cols*5+1+ploti)
            ax.plot(routs[:,self.Nsub*4:self.Nsub*4+10],
                        linewidth=pltu.plot_linewidth)
            ax.set_ylim([-0.1,1.1])
            pltu.beautify_plot(ax,x0min=False,y0min=False,
                    xticks=(),yticks=yticks)
            pltu.axes_labels(ax,'',ylabels[5])
        ax = fig.add_subplot(8,cols,cols*6+1+ploti)
        ax.plot(MDouts,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=(),yticks=yticks)
        pltu.axes_labels(ax,'',ylabels[6])
        ax = fig.add_subplot(8,cols,cols*7+1+ploti)
        ax.plot(outs,linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1,1.1])
        pltu.beautify_plot(ax,x0min=False,y0min=False,
                xticks=[0,self.tsteps],yticks=yticks)
        pltu.axes_labels(ax,'time (ms)',ylabels[7])
        fig.tight_layout()
        
        if self.saveData:
            d = {}
            # 1st column of all matrices is number of time steps
            # 2nd column is number of neurons / units
            d['cues'] = cues                # tsteps x 4
            d['routs'] = routs              # tsteps x 1000
            d['MDouts'] = MDouts            # tsteps x 2
            d['outs'] = outs                # tsteps x 2
            sio.savemat('simData'+str(ploti)+'.mat', d)
        
        return ax

    def performance(self,cuei,outs,errors,target):
        meanErr = np.mean(errors[-100:,:]*errors[-100:,:])
        # endout is the mean of all end 100 time points for each output
        endout = np.mean(outs[-100:,:],axis=0)
        targeti = 0 if target[0]>target[1] else 1
        non_targeti = 1 if target[0]>target[1] else 0
        ## endout for targeti output must be greater than for the other
        ##  with a margin of 50% of desired difference of 1. between the two
        #if endout[targeti] > (endout[non_targeti]+0.5): correct = 1
        #else: correct = 0
        # just store the margin of error instead of thresholding it
        correct = endout[targeti] - endout[non_targeti]
        return meanErr, correct

    def do_test(self,Ntest,cueList,cuePlot,colNum,train=True):
        NcuesTest = len(cueList)
        MSEs = np.zeros(Ntest*NcuesTest)
        corrects = np.zeros(Ntest*NcuesTest)
        wOuts = np.zeros((Ntest,self.Nout,self.Nneur))
        self.meanAct = np.zeros(shape=(self.Ntasks*self.inpsPerTask,\
                                        self.tsteps,self.Nneur))
        #
        for testi in range(Ntest):
            if self.plotFigs: print('Simulating test cycle',testi)
            cues_order = self.get_cues_order(cueList)
            for cuenum,(taski,cuei) in enumerate(cues_order):
                cue, target = self.get_cue_target(taski,cuei)
                cues, routs, outs, MDouts, errors = \
                    self.sim_cue(taski,cuei,cue,target,train=train)
                MSEs[testi*NcuesTest+cuenum], corrects[testi*NcuesTest+cuenum] = \
                    self.performance(cuei,outs,errors,target)

                if cuePlot is not None:
                    if self.plotFigs and testi == 0 and taski==cuePlot[0] and cuei==cuePlot[1]:
                        ax = self.plot_column(self.fig,cues,routs,MDouts,outs,ploti=colNum)
            
            wOuts[testi,:,:] = self.wOut

        self.meanAct /= Ntest
        if self.plotFigs and cuePlot is not None:
            ax.text(0.1,0.4,'{:1.2f}$\pm${:1.2f}'.format(np.mean(corrects),np.std(corrects)),
                        transform=ax.transAxes)
            ax.text(0.1,0.25,'{:1.2f}$\pm${:1.2f}'.format(np.mean(MSEs),np.std(MSEs)),
                        transform=ax.transAxes)

        if self.saveData:
            # 1-Dim: numCycles * 4 cues/cycle i.e. 70*4=280
            self.fileDict['corrects'+str(colNum)] = corrects
            # at each cycle, a weights matrix 2x1000:
            # weights to 2 output neurons from 1000 cue-selective neurons
            # 3-Dim: 70 (numCycles) x 2 x 1000
            self.fileDict['wOuts'+str(colNum)] = wOuts
            #savemat('simDataTrials'+str(colNum), d)
        
        return MSEs,corrects,wOuts

    def get_cue_list(self,taski=None):
        if taski is not None:
            # (taski,cuei) combinations for one given taski
            cueList = np.dstack(( np.repeat(taski,self.inpsPerTask),
                                    np.arange(self.inpsPerTask) ))
        else:
            # every possible (taski,cuei) combination
            cueList = np.dstack(( np.repeat(np.arange(self.Ntasks),self.inpsPerTask),
                                    np.tile(np.arange(self.inpsPerTask),self.Ntasks) ))
        return cueList[0]

    def train(self,Ntrain):
        if self.blockTrain:
            Nextra = 200            # add cycles to show if block1 learning is remembered
            Ntrain = Ntrain*self.Ntasks + Nextra
        else:
            Ntrain *= self.Ntasks
        wOuts = np.zeros(shape=(Ntrain,self.Nout,self.Nneur))
        
        # Reset the trained weights,
        #  earlier for iterating over MDeffect = False and then True
        self.wOut = np.random.uniform(-1,1,
                        size=(self.Nout,self.Nneur))/self.Nneur

        MSEs = np.zeros(Ntrain)
        for traini in range(Ntrain):
            
            if self.plotFigs: print('Simulating training cycle',traini)
            
            ## reduce learning rate by *10 from 100th and 200th cycle
            #if traini == 100: self.learning_rate /= 10.
            #elif traini == 200: self.learning_rate /= 10.
            
            # if blockTrain,
            #  first half of trials is context1, second half is context2
            if self.blockTrain:
                taski = traini // ((Ntrain-Nextra)//self.Ntasks)
                # last block is just the first context again
                if traini >= Ntrain-Nextra: taski = 0
                cueList = self.get_cue_list(taski)
            else:
                cueList = self.get_cue_list()
            cues_order = self.get_cues_order(cueList)
            for taski,cuei in cues_order:
                cue, target = \
                    self.get_cue_target(taski,cuei)
                cues, routs, outs, MDouts, errors = \
                    self.sim_cue(taski,cuei,cue,target,train=True)

                MSEs[traini] += np.mean(errors*errors)
                
                wOuts[traini,:,:] = self.wOut
        self.meanAct /= Ntrain

        if self.saveData:
            self.fileDict['MSEs'] = MSEs
            self.fileDict['wOuts'] = wOuts
            
            pickle_out = open('dataPFCMD/train_'+\
                                    str(self.MDstrength)+\
                                    '_R'+str(self.RNGSEED)+\
                                    ('_xor' if self.xorTask else '')+'.pickle','wb')
            pickle.dump({'MSEs':MSEs, 'wOut':wOuts},pickle_out)
            pickle_out.close()
            
            sio.savemat('dataPFCMD/train_'+\
                                    str(self.MDstrength)+\
                                    '_R'+str(self.RNGSEED)+\
                                    ('_xor' if self.xorTask else '')+'.mat', {'MSEs':MSEs, 'wOut':wOuts})

        if self.plotFigs:
            self.fig2 = plt.figure(
                            figsize=(pltu.columnwidth,pltu.columnwidth),
                            facecolor='w')
            ax2 = self.fig2.add_subplot(1,1,1)
            ax2.plot(MSEs)
            pltu.beautify_plot(ax2,x0min=False,y0min=False)
            pltu.axes_labels(ax2,'cycle num','MSE')
            self.fig2.tight_layout()

            # plot output weights evolution
            self.fig3 = plt.figure(
                            figsize=(pltu.columnwidth,pltu.columnwidth),
                            facecolor='w')
            ax3 = self.fig3.add_subplot(2,1,1)
            ax3.plot(wOuts[:,0,:5],'-,r')
            ax3.plot(wOuts[:,1,:5],'-,b')
            pltu.beautify_plot(ax3,x0min=False,y0min=False)
            pltu.axes_labels(ax3,'cycle num','wAto0(r) wAto1(b)')
            ax4 = self.fig3.add_subplot(2,1,2)
            ax4.plot(wOuts[:,0,self.Nsub:self.Nsub+5],'-,r')
            ax4.plot(wOuts[:,1,self.Nsub:self.Nsub+5],'-,b')
            pltu.beautify_plot(ax4,x0min=False,y0min=False)
            pltu.axes_labels(ax4,'cycle num','wBto0(r) wBto1(b)')
            self.fig3.tight_layout()

    def test(self,Ntest):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
            self.fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
        cues = self.get_cue_list()
        
        # after learning, during testing the learning rate is low, just performance tuning
        self.learning_rate /= 100.
        #import pdb;pdb.set_trace()
        self.do_test(Ntest,cues,(0,0),0)
        if self.plotFigs:
            ax = self.fig2.add_subplot(111)
            # plot mean activity of each neuron for this taski+cuei
            #  further binning 10 neurons into 1
            ax.plot(np.mean(np.reshape(\
                                np.mean(self.meanAct[0,:,:],axis=0),\
                            (self.Nneur//10,10)),axis=1),',-r')
        if self.saveData:
            self.fileDict['meanAct0'] = self.meanAct[0,:,:]
        self.do_test(Ntest,cues,(0,1),1)
        if self.plotFigs:
            # plot mean activity of each neuron for this taski+cuei
            ax.plot(np.mean(np.reshape(\
                                np.mean(self.meanAct[1,:,:],axis=0),\
                            (self.Nneur//10,10)),axis=1),',-b')
            ax.set_xlabel('neuron #')
            ax.set_ylabel('mean rate')
        if self.saveData:
            self.fileDict['meanAct1'] = self.meanAct[1,:,:]

        if self.xorTask:
            self.do_test(Ntest,cues,(0,2),2)
            self.do_test(Ntest,cues,(0,3),3)
        else:
            self.do_test(Ntest,cues,(1,0),2)
            self.do_test(Ntest,cues,(1,1),3)
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('fig_plasticPFC2Out.png',
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            self.fig2.tight_layout()

    def load(self,filename):
        d = shelve.open(filename) # open
        self.wOut = d['wOut']
        d.close()
        return None

    def save(self):
        self.fileDict['wOut'] = self.wOut
        
        

if __name__ == "__main__":
    #PFC_G = 1.6                    # if not positiveRates
    PFC_G = 6.
    PFC_G_off = 1.5
    learning_rate = 5e-6
    learning_cycles_per_task = 1000
    Ntest = 20
    Nblock = 70
    noiseSD = 1e-3
    tauError = 0.001
    reLoadWeights = False
    saveData = not reLoadWeights
    plotFigs = True
    pfcmd = PFCMD(PFC_G,PFC_G_off,learning_rate,
                    noiseSD,tauError,plotFigs=plotFigs,saveData=saveData)
    if not reLoadWeights:
        pfcmd.train(learning_cycles_per_task)
        if saveData:
            pfcmd.save()
        # save weights right after training,
        #  since test() keeps training on during MD off etc.
        #import pdb;pdb.set_trace()
        #pfcmd.test(Ntest)
    else:
        pfcmd.load(filename)
        # all 4cues in a block
        pfcmd.test(Ntest)

    if pfcmd.saveData:
        pfcmd.fileDict.close()
    
    plt.show()
