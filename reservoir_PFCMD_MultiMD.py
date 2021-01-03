# -*- coding: utf-8 -*-
# (c) May 2018 Aditya Gilra, EPFL.

"""Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import sys,shelve
import plot_utils as pltu
import pickle

class PFCMD():
    def __init__(self,PFC_G,PFC_G_off,learning_rate,
                    noiseSD,tauError,plotFigs=True,saveData=False):
        self.RNGSEED = 5
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
        
        self.tau_times = 4
        self.Num_MD = 10
        self.learning_rate = learning_rate  # too high a learning rate makes the output weights
                                            #  change too much within a trial / training cycle,
                                            #  then the output interference depends
                                            #  on the order of cues within a cycle
                                            # typical values is 1e-5, can vary from 1e-4 to 1e-6
        self.tauError = tauError            # smooth the error a bit, so that weights don't fluctuate

        self.MDeffect = True                # whether to have MD present or not
        self.MDEffectType = 'submult'       # MD subtracts from across tasks and multiplies within task
        #self.MDEffectType = 'subadd'        # MD subtracts from across tasks and adds within task
        #self.MDEffectType = 'divadd'        # MD divides from across tasks and adds within task
        #self.MDEffectType = 'divmult'       # MD divides from across tasks and multiplies within task

        self.dirConn = False                # direct connections from cue to output, also learned
        self.outExternal = True             # True: output neurons are external to the PFC
                                            #  (i.e. weights to and fro (outFB) are not MD modulated)
                                            # False: last self.Nout neurons of PFC are output neurons
        self.outFB = False                  # if outExternal, then whether feedback from output to reservoir
        self.noisePresent = False           # add noise to all reservoir units

        self.positiveRates = True           # whether to clip rates to be only positive, G must also change
        
        self.MDlearn = True# False                # whether MD should learn
                                            #  possibly to make task representations disjoint (not just orthogonal)

        #self.MDstrength = None              # if None, use wPFC2MD, if not None as below, just use context directly
        #self.MDstrength = 0.                # a parameter that controls how much the MD disjoints task representations.
        self.MDstrength = 1.                # a parameter that controls how much the MD disjoints task representations.
                                            #  zero would be a pure reservoir, 1 would be full MDeffect
                                            # -1 for zero recurrent weights
        self.wInSpread = False              # Spread wIn also into other cue neurons to see if MD disjoints representations
        self.blockTrain = True              # first half of training is context1, second half is context2
        
        self.reinforce = False              # use reinforcement learning (node perturbation) a la Miconi 2017
                                            #  instead of error-driven learning
                                            
        if self.reinforce:
            self.perturbProb = 50./self.tsteps
                                            # probability of perturbation of each output neuron per time step
            self.perturbAmpl = 10.          # how much to perturb the output by
            self.meanErrors = np.zeros(self.Ntasks*self.inpsPerTask)
                                            # vector holding running mean error for each cue
            self.decayErrorPerTrial = 0.1   # how to decay the mean errorEnd by, per trial
            self.learning_rate *= 10        # increase learning rate for reinforce
            self.reinforceReservoir = False # learning on reservoir weights also?
            if self.reinforceReservoir:
                self.perturbProb /= 10

        self.depress = False                # a depressive term if there is pre-post firing
        self.multiAttractorReservoir = False# increase the reservoir weights within each cue
                                            #  all uniformly (could also try Hopfield style for the cue pattern)
        if self.outExternal:
            self.wOutMask = np.ones(shape=(self.Nout,self.Nneur))
            #self.wOutMask[ np.random.uniform( \
            #            size=(self.Nout,self.Nneur)) > 0.3 ] = 0.
            #                                # output weights sparsity, 30% sparsity

        self.wPFC2MD = np.zeros(shape=(self.Num_MD,self.Nneur))
        
        if self.MDEffectType == 'submult':
            # working!
            Gbase = 0.75                      # determines also the cross-task recurrence
            if self.MDstrength is None: MDval = 1.
            elif self.MDstrength < 0.: MDval = 0.
            else: MDval = self.MDstrength
            # subtract across tasks (task with higher MD suppresses cross-tasks)
            self.wMD2PFC = np.ones(shape=(self.Nneur,self.Num_MD)) * (-10.) * MDval
            self.useMult = True
            # multiply recurrence within task, no addition across tasks
            ## choose below option for cross-recurrence
            ##  if you want "MD inactivated" (low recurrence) state
            ##  as the state before MD learning
            #self.wMD2PFCMult = np.zeros(shape=(self.Nneur,self.Ntasks))
            # choose below option for cross-recurrence
            #  if you want "reservoir" (high recurrence) state
            #  as the state before MD learning (makes learning more difficult)
            self.wMD2PFCMult = np.ones(shape=(self.Nneur,self.Num_MD)) \
                                * PFC_G_off/Gbase * (1-MDval)
            # threshold for sharp sigmoid (0.1 width) transition of MDinp
            self.MDthreshold = 0.4
        else:
            #print 'undefined inhibitory effect of MD'
            sys.exit(1)
        # With MDeffect = True and MDstrength = 0, i.e. MD inactivated
        #  PFC recurrence is (1+PFC_G_off)*Gbase = (1+1.5)*0.75 = 1.875
        # So with MDeffect = False, ensure the same PFC recurrence for the pure reservoir
        if not self.MDeffect: Gbase = 1.875

        if self.MDeffect and self.MDlearn:
            self.wPFC2MD = np.random.normal(size=(self.Num_MD,self.Nneur))
            self.wMD2PFC = np.random.normal(size=(self.Nneur,self.Num_MD))
            self.wMD2PFCMult = np.random.normal(size=(self.Nneur,self.Num_MD))
            self.wMD2PFC *= 0.
            self.wMD2PFCMult *= 0.
            self.MDpreTrace = np.zeros(shape=(self.Nneur))

        # Choose G based on the type of activation function
        # unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if self.positiveRates:
            self.G = Gbase
            self.tauMD = self.tau*self.tau_times ##self.tau
        else:
            self.G = Gbase
            self.MDthreshold = 0.4
            self.tauMD = self.tau*10*self.tau_times

        # Perhaps I shouldn't have self connections / autapses?!
        # Perhaps I should have sparse connectivity?
        self.Jrec = np.random.normal(size=(self.Nneur, self.Nneur))\
                        *self.G/np.sqrt(self.Nsub*2)
        if self.MDstrength < 0.: self.Jrec *= 0.
        if self.multiAttractorReservoir:
            for i in range(self.Ncues):
                self.Jrec[self.Nsub*i:self.Nsub*(i+1)] *= 2.

        # make mean input to each row zero,
        #  helps to avoid saturation (both sides) for positive-only rates.
        #  see Nicola & Clopath 2016
        # mean of rows i.e. across columns (axis 1),
        #  then expand with np.newaxis
        #   so that numpy's broadcast works on rows not columns
        self.Jrec -= np.mean(self.Jrec,axis=1)[:,np.newaxis]
        #for i in range(self.Nsub):
        #    self.Jrec[i,:self.Nsub] -= np.mean(self.Jrec[i,:self.Nsub])
        #    self.Jrec[self.Nsub+i,self.Nsub:self.Nsub*2] -=\
        #        np.mean(self.Jrec[self.Nsub+i,self.Nsub:self.Nsub*2])
        #    self.Jrec[self.Nsub*2+i,self.Nsub*2:self.Nsub*3] -=\
        #        np.mean(self.Jrec[self.Nsub*2+i,self.Nsub*2:self.Nsub*3])
        #    self.Jrec[self.Nsub*3+i,self.Nsub*3:self.Nsub*4] -=\
        #        np.mean(self.Jrec[self.Nsub*3+i,self.Nsub*3:self.Nsub*4])

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

        #wIn = np.random.uniform(-1,1,size=(self.Nneur,self.Ncues))
        self.wIn = np.zeros((self.Nneur,self.Ncues))
        self.cueFactor = 1.5
        if self.positiveRates: lowcue,highcue = 0.5,1.
        else: lowcue,highcue = -1.,1
        for cuei in np.arange(self.Ncues):
            self.wIn[self.Nsub*cuei:self.Nsub*(cuei+1),cuei] = \
                    np.random.uniform(lowcue,highcue,size=self.Nsub) \
                            *self.cueFactor
            if self.wInSpread:
                # small cross excitation to half the neurons of cue-1 (wrap-around)
                if cuei == 0: endidx = self.Nneur
                else: endidx = self.Nsub*cuei
                self.wIn[self.Nsub*cuei - self.Nsub//2 : endidx,cuei] += \
                        np.random.uniform(0.,lowcue,size=self.Nsub//2) \
                                *self.cueFactor
                # small cross excitation to half the neurons of cue+1 (wrap-around)
                self.wIn[(self.Nsub*(cuei+1))%self.Nneur : \
                            (self.Nsub*(cuei+1) + self.Nsub//2 )%self.Nneur,cuei] += \
                        np.random.uniform(0.,lowcue,size=self.Nsub//2) \
                                *self.cueFactor

        # wDir and wOut are set in the main training loop
        if self.outExternal and self.outFB:
            self.wFB = np.random.uniform(-1,1,size=(self.Nneur,self.Nout))\
                            *self.G/np.sqrt(self.Nsub*2)*PFC_G

        self.cue_eigvecs = np.zeros((self.Ncues,self.Nneur))
        self.plotFigs = plotFigs
        self.cuePlot = (0,0)
                
#        if self.saveData:
#            self.fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD'+\
#                                    str(self.MDstrength)+\
#                                    '_R'+str(self.RNGSEED)+\
#                                    ('_xor' if self.xorTask else '')+'.shelve')

        self.meanAct = np.zeros(shape=(self.Ntasks*self.inpsPerTask,\
                                    self.tsteps,self.Nneur))

    def sim_cue(self,taski,cuei,cue,target,MDeffect=True,
                    MDCueOff=False,MDDelayOff=False,
                    train=True,routsTarget=None):
        '''
        self.reinforce trains output weights
         using REINFORCE / node perturbation a la Miconi 2017.'''
        cues = np.zeros(shape=(self.tsteps,self.Ncues))
        # random initialization of input to units
        # very important to have some random input
        #  just for the xor task for (0,0) cue!
        #  keeping it also for the 1:1 task just for consistency
        xinp = np.random.uniform(0,0.1,size=(self.Nneur))
        #xinp = np.zeros(shape=(self.Nneur))
        xadd = np.zeros(shape=(self.Nneur))
        MDinp = np.zeros(shape=self.Num_MD)
        routs = np.zeros(shape=(self.tsteps,self.Nneur))
        MDouts = np.zeros(shape=(self.tsteps,self.Num_MD))
        MDinps = np.zeros(shape=(self.tsteps,self.Num_MD))
        outInp = np.zeros(shape=self.Nout)
        outs = np.zeros(shape=(self.tsteps,self.Nout))
        out = np.zeros(self.Nout)
        errors = np.zeros(shape=(self.tsteps,self.Nout))
        error_smooth = np.zeros(shape=self.Nout)
        if self.reinforce:
            HebbTrace = np.zeros(shape=(self.Nout,self.Nneur))
            if self.dirConn:
                HebbTraceDir = np.zeros(shape=(self.Nout,self.Ncues))
            if self.reinforceReservoir:
                HebbTraceRec = np.zeros(shape=(self.Nneur,self.Nneur))

        for i in range(self.tsteps):
            rout = self.activation(xinp)
            routs[i,:] = rout
            #import pdb;pdb.set_trace()
            if self.outExternal:
                outAdd = np.dot(self.wOut,rout)

            if MDeffect:
                # MD decays 10x slower than PFC neurons,
                #  so as to somewhat integrate PFC input
                if self.positiveRates:
                    MDinp +=  self.dt/self.tauMD * \
                            ( -MDinp + np.dot(self.wPFC2MD,rout) )
                else: # shift PFC rates, so that mean is non-zero to turn MD on
                    MDinp +=  self.dt/self.tauMD * \
                            ( -MDinp + np.dot(self.wPFC2MD,(rout+1./2)) )

                # MD off during cue or delay periods:
                if MDCueOff and i<self.cuesteps:
                    MDinp = np.zeros(self.Num_MD)
                    #MDout /= 2.
                if MDDelayOff and i>self.cuesteps and i<self.tsteps:
                    MDinp = np.zeros(self.Num_MD)

                # MD out either from MDinp or forced
#                if self.MDstrength is not None:
#                    MDout = np.zeros(self.Ntasks)
#                    MDout[taski] = 1.
#                else:
#                    MDout = (np.tanh( (MDinp-self.MDthreshold)/0.1 ) + 1) / 2.
                # if MDlearn then force "winner take all" on MD output
                if train and self.MDlearn:
                    MDout = (np.tanh(MDinp-self.MDthreshold) + 1) / 2.
                    # winner take all on the MD
                    #  hardcoded for self.Ntasks = 2
#                    if MDinp[0] > MDinp[1]: MDout = np.array([1,0])
#                    else: MDout = np.array([0,1])

                MDouts[i,:] = MDout
                MDinps[i,:] = MDinp

                if self.useMult:
                    self.MD2PFCMult = np.dot(self.wMD2PFCMult,MDout)
                    xadd = (1.+self.MD2PFCMult) * np.dot(self.Jrec,rout)
                else:
                    xadd = np.dot(self.Jrec,rout)
                xadd += np.dot(self.wMD2PFC,MDout)

                if train and self.MDlearn:
                    # MD presynaptic traces filtered over 10 trials
                    # Ideally one should weight them with MD syn weights,
                    #  but syn plasticity just uses pre!
                    self.MDpreTrace += 1./self.tsteps/10. * \
                                        ( -self.MDpreTrace + rout )
                    wPFC2MDdelta = 1e-4*np.outer(MDout-0.5,self.MDpreTrace-0.13)
                    self.wPFC2MD = np.clip(self.wPFC2MD+wPFC2MDdelta,0.,1.)
                    self.wMD2PFC = np.clip(self.wMD2PFC+wPFC2MDdelta.T,-10.,0.)
                    self.wMD2PFCMult = np.clip(self.wMD2PFCMult+wPFC2MDdelta.T,0.,7./self.G)
            else:
                xadd = np.dot(self.Jrec,rout)

            if i < self.cuesteps:
                ## add an MDeffect on the cue
                #if MDeffect and useMult:
                #    xadd += self.MD2PFCMult * np.dot(self.wIn,cue)
                # baseline cue is always added
                xadd += np.dot(self.wIn,cue)
                cues[i,:] = cue
                if self.dirConn:
                    if self.outExternal:
                        outAdd += np.dot(self.wDir,cue)
                    else:
                        xadd[-self.Nout:] += np.dot(self.wDir,cue)

            if self.reinforce:
                # Exploratory perturbations a la Miconi 2017
                # Perturb each output neuron independently
                #  with probability perturbProb
                perturbationOff = np.where(
                        np.random.uniform(size=self.Nout)>=self.perturbProb )
                perturbation = np.random.uniform(-1,1,size=self.Nout)
                perturbation[perturbationOff] = 0.
                perturbation *= self.perturbAmpl
                outAdd += perturbation
            
                if self.reinforceReservoir:
                    perturbationOff = np.where(
                            np.random.uniform(size=self.Nneur)>=self.perturbProb )
                    perturbationRec = np.random.uniform(-1,1,size=self.Nneur)
                    perturbationRec[perturbationOff] = 0.
                    # shouldn't have MD mask on perturbations,
                    #  else when MD is off, perturbations stop!
                    #  use strong subtractive inhibition to kill perturbation
                    #   on task irrelevant neurons when MD is on.
                    #perturbationRec *= self.MD2PFCMult  # perturb gated by MD
                    perturbationRec *= self.perturbAmpl
                    xadd += perturbationRec

            if self.outExternal and self.outFB:
                xadd += np.dot(self.wFB,out)
            xinp += self.dt/self.tau * (-xinp + xadd)
            
            if self.noisePresent:
                xinp += np.random.normal(size=(self.Nneur))*self.noiseSD \
                            * np.sqrt(self.dt)/self.tau
            
            if self.outExternal:
                outInp += self.dt/self.tau * (-outInp + outAdd)
                out = self.activation(outInp)                
            else:
                out = rout[-self.Nout:]
            error = out - target
            errors[i,:] = error
            outs[i,:] = out
            error_smooth += self.dt/self.tauError * (-error_smooth + error)
            
            if train:
                if self.reinforce:
                    # note: rout is the activity vector for previous time step
                    HebbTrace += np.outer(perturbation,rout)
                    if self.dirConn:
                        HebbTraceDir += np.outer(perturbation,cue)
                    if self.reinforceReservoir:
                        HebbTraceRec += np.outer(perturbationRec,rout)
                else:
                    # error-driven i.e. error*pre (perceptron like) learning
                    if self.outExternal:
                        self.wOut += -self.learning_rate \
                                        * np.outer(error_smooth,rout)
                        if self.depress:
                            self.wOut -= 10*self.learning_rate \
                                        * np.outer(out,rout)*self.wOut
                    else:
                        self.Jrec[-self.Nout:,:] += -self.learning_rate \
                                        * np.outer(error_smooth,rout)
                        if self.depress:
                            self.Jrec[-self.Nout:,:] -= 10*self.learning_rate \
                                        * np.outer(out,rout)*self.Jrec[-self.Nout:,:]
                    if self.dirConn:
                        self.wDir += -self.learning_rate \
                                        * np.outer(error_smooth,cue)
                        if self.depress:
                            self.wDir -= 10*self.learning_rate \
                                        * np.outer(out,cue)*self.wDir

        inpi = taski*self.inpsPerTask + cuei
        if train and self.reinforce:
            # with learning using REINFORCE / node perturbation (Miconi 2017),
            #  the weights are only changed once, at the end of the trial
            # apart from eta * (err-baseline_err) * hebbianTrace,
            #  the extra factor baseline_err helps to stabilize learning
            #   as per Miconi 2017's code,
            #  but I found that it destabilized learning, so not using it.
            errorEnd = np.mean(errors*errors)
            if self.outExternal:
                self.wOut -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTrace #* self.meanErrors[inpi]
            else:
                self.Jrec[-self.Nout:,:] -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTrace #* self.meanErrors[inpi]
            if self.reinforceReservoir:
                self.Jrec -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceRec #* self.meanErrors[inpi]                
            if self.dirConn:
                sefl.wDir -= self.learning_rate * \
                        (errorEnd-self.meanErrors[inpi]) * \
                            HebbTraceDir #* self.meanErrors[inpi]
        
            # cue-specific mean error (low-pass over many trials)
            self.meanErrors[inpi] = \
                self.decayErrorPerTrial * self.meanErrors[inpi] + \
                (1.0 - self.decayErrorPerTrial) * errorEnd

        if train and self.outExternal:
            self.wOut *= self.wOutMask
        
        self.meanAct[inpi,:,:] += routs

        return cues, routs, outs, MDouts, errors, MDinps

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
            savemat('simData'+str(ploti), d)
        
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
    
    def test_new(self,Ntest):
        MDeffect = self.MDeffect
        Ntest *= self.Ntasks
        wOuts = np.zeros(shape=(Ntest,self.Nout,self.Nneur))
        
        cues_all = np.zeros(shape=(Ntest,self.tsteps,self.Ncues))
        routs_all = np.zeros(shape=(Ntest,self.tsteps,self.Nneur))
        MDouts_all = np.zeros(shape=(Ntest,self.tsteps,self.Num_MD))
        MDinps_all = np.zeros(shape=(Ntest,self.tsteps,self.Num_MD))
        outs_all = np.zeros(shape=(Ntest,self.tsteps,self.Nout))
        
        MSEs = np.zeros(Ntest)
        for testi in range(Ntest):

            cueList = self.get_cue_list()
            cues_order = self.get_cues_order(cueList)
            for taski,cuei in cues_order:
                cue, target = \
                    self.get_cue_target(taski,cuei)
                cues, routs, outs, MDouts, errors, MDinps = \
                    self.sim_cue(taski,cuei,cue,target,MDeffect=MDeffect,
                    train=True)

                MSEs[testi] += np.mean(errors*errors)
                
                wOuts[testi,:,:] = self.wOut
                
                cues_all[testi,:,:] = cues
                routs_all[testi,:,:] = routs
                MDouts_all[testi,:,:] = MDouts
                MDinps_all[testi,:,:] = MDinps
                outs_all[testi,:,:] = outs

        self.meanAct /= Ntest

        if self.saveData:
#            self.fileDict['MSEs'] = MSEs
#            self.fileDict['wOuts'] = wOuts
            
            pickle_out = open('dataPFCMD/test_numMD'+str(self.Num_MD)+'_MD'+\
                                    str(self.MDeffect)+\
                                    '_Learn'+str(self.MDlearn)+\
                                    '_R'+str(self.RNGSEED)+\
                                    '_TimesTau'+str(self.tau_times)+\
                                    '.pickle','wb')
            pickle.dump({'MSEs':MSEs, 'cues_all':cues_all,'routs_all':routs_all,\
                         'MDouts_all':MDouts_all,'MDinps_all':MDinps_all,'outs_all':outs_all},pickle_out)
            pickle_out.close()
            
    def do_test(self,Ntest,MDeffect,MDCueOff,MDDelayOff,
                    cueList,cuePlot,colNum,train=True):
        NcuesTest = len(cueList)
        MSEs = np.zeros(Ntest*NcuesTest)
        corrects = np.zeros(Ntest*NcuesTest)
        wOuts = np.zeros((Ntest,self.Nout,self.Nneur))
        self.meanAct = np.zeros(shape=(self.Ntasks*self.inpsPerTask,\
                                        self.tsteps,self.Nneur))
        for testi in range(Ntest):
            if self.plotFigs: print('Simulating test cycle',testi)
            cues_order = self.get_cues_order(cueList)
            for cuenum,(taski,cuei) in enumerate(cues_order):
                cue, target = self.get_cue_target(taski,cuei)
                cues, routs, outs, MDouts, errors, MDinps = \
                    self.sim_cue(taski,cuei,cue,target,
                            MDeffect,MDCueOff,MDDelayOff,train=train)
                MSEs[testi*NcuesTest+cuenum], corrects[testi*NcuesTest+cuenum] = \
                    self.performance(cuei,outs,errors,target)

                if cuePlot is not None:
                    if self.plotFigs and testi == 0 and taski==cuePlot[0] and cuei==cuePlot[1]:
                        ax = self.plot_column(self.fig,cues,routs,MDouts,outs,ploti=colNum)

            if self.outExternal:
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
        MDeffect = self.MDeffect
        if self.blockTrain:
            Nextra = 200            # add cycles to show if block1 learning is remembered
            Ntrain = Ntrain*self.Ntasks + Nextra
        else:
            Ntrain *= self.Ntasks
        wOuts = np.zeros(shape=(Ntrain,self.Nout,self.Nneur))
        
        cues_all = np.zeros(shape=(Ntrain,self.tsteps,self.Ncues))
        routs_all = np.zeros(shape=(Ntrain,self.tsteps,self.Nneur))
        MDouts_all = np.zeros(shape=(Ntrain,self.tsteps,self.Num_MD))
        MDinps_all = np.zeros(shape=(Ntrain,self.tsteps,self.Num_MD))
        outs_all = np.zeros(shape=(Ntrain,self.tsteps,self.Nout))
        
        
        
        if self.MDlearn:
            wPFC2MDs = np.zeros(shape=(Ntrain,self.Num_MD,self.Nneur))
            wMD2PFCs = np.zeros(shape=(Ntrain,self.Nneur,self.Num_MD))
            wMD2PFCMults = np.zeros(shape=(Ntrain,self.Nneur,self.Num_MD))
            MDpreTraces = np.zeros(shape=(Ntrain,self.Nneur))
        
        # Reset the trained weights,
        #  earlier for iterating over MDeffect = False and then True
        if self.outExternal:
            self.wOut = np.random.uniform(-1,1,
                            size=(self.Nout,self.Nneur))/self.Nneur
            self.wOut *= self.wOutMask
        elif not MDeffect:
            self.Jrec[-self.Nout:,:] = \
                np.random.normal(size=(self.Nneur, self.Nneur))\
                            *self.G/np.sqrt(self.Nsub*2)
        # direct connections from cue to output,
        #  similar to having output neurons within reservoir
        if self.dirConn:
            self.wDir = np.random.uniform(-1,1,
                            size=(self.Nout,self.Ncues))\
                            /self.Ncues *1.5

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
                cues, routs, outs, MDouts, errors, MDinps = \
                    self.sim_cue(taski,cuei,cue,target,MDeffect=MDeffect,
                    train=True)

                MSEs[traini] += np.mean(errors*errors)
                
                wOuts[traini,:,:] = self.wOut
                
                cues_all[traini,:,:] = cues
                routs_all[traini,:,:] = routs
                MDouts_all[traini,:,:] = MDouts
                MDinps_all[traini,:,:] = MDinps
                outs_all[traini,:,:] = outs
                
                if self.plotFigs and self.outExternal:
                    if self.MDlearn:
                        wPFC2MDs[traini,:,:] = self.wPFC2MD
                        wMD2PFCs[traini,:,:] = self.wMD2PFC
                        wMD2PFCMults[traini,:,:] = self.wMD2PFCMult
                        MDpreTraces[traini,:] = self.MDpreTrace
        self.meanAct /= Ntrain

#        if self.saveData:
##            self.fileDict['MSEs'] = MSEs
##            self.fileDict['wOuts'] = wOuts
#            
#            pickle_out = open('dataPFCMD/activity_noise_MD'+\
#                                    str(self.MDeffect)+\
#                                    '_Learn'+str(self.MDlearn)+\
#                                    '_R'+str(self.RNGSEED)+\
#                                    '_TimesTau'+str(self.tau_times)+\
#                                    '.pickle','wb')
#            pickle.dump({'MSEs':MSEs, 'cues_all':cues_all,'routs_all':routs_all,\
#                         'MDouts_all':MDouts_all,'MDinps_all':MDinps_all,'outs_all':outs_all},pickle_out)
#            pickle_out.close()


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

#            if self.MDlearn:
#                # plot PFC2MD weights evolution
#                self.fig3 = plt.figure(
#                                figsize=(pltu.columnwidth,pltu.columnwidth),
#                                facecolor='w')
#                ax3 = self.fig3.add_subplot(2,1,1)
#                ax3.plot(wPFC2MDs[:,0,:5],'-,r')
#                ax3.plot(wPFC2MDs[:,0,self.Nsub*2:self.Nsub*2+5],'-,b')
#                pltu.beautify_plot(ax3,x0min=False,y0min=False)
#                pltu.axes_labels(ax3,'cycle num','wAtoMD0(r) wCtoMD0(b)')
#                ax4 = self.fig3.add_subplot(2,1,2)
#                ax4.plot(wPFC2MDs[:,1,:5],'-,r')
#                ax4.plot(wPFC2MDs[:,1,self.Nsub*2:self.Nsub*2+5],'-,b')
#                pltu.beautify_plot(ax4,x0min=False,y0min=False)
#                pltu.axes_labels(ax4,'cycle num','wAtoMD1(r) wCtoMD1(b)')
#                self.fig3.tight_layout()
#
#                # plot MD2PFC weights evolution
#                self.fig3 = plt.figure(
#                                figsize=(pltu.columnwidth,pltu.columnwidth),
#                                facecolor='w')
#                ax3 = self.fig3.add_subplot(2,1,1)
#                ax3.plot(wMD2PFCs[:,:5,0],'-,r')
#                ax3.plot(wMD2PFCs[:,self.Nsub*2:self.Nsub*2+5,0],'-,b')
#                pltu.beautify_plot(ax3,x0min=False,y0min=False)
#                pltu.axes_labels(ax3,'cycle num','wMD0toA(r) wMD0toC(b)')
#                ax4 = self.fig3.add_subplot(2,1,2)
#                ax4.plot(wMD2PFCMults[:,:5,0],'-,r')
#                ax4.plot(wMD2PFCMults[:,self.Nsub*2:self.Nsub*2+5,0],'-,b')
#                pltu.beautify_plot(ax4,x0min=False,y0min=False)
#                pltu.axes_labels(ax4,'cycle num','MwMD0toA(r) MwMD0toC(b)')
#                self.fig3.tight_layout()
#
#                # plot PFC to MD pre Traces
#                self.fig3 = plt.figure(
#                                figsize=(pltu.columnwidth,pltu.columnwidth),
#                                facecolor='w')
#                ax3 = self.fig3.add_subplot(1,1,1)
#                ax3.plot(MDpreTraces[:,:5],'-,r')
#                ax3.plot(MDpreTraces[:,self.Nsub*2:self.Nsub*2+5],'-,b')
#                pltu.beautify_plot(ax3,x0min=False,y0min=False)
#                pltu.axes_labels(ax3,'cycle num','cueApre(r) cueCpre(b)')
#                self.fig3.tight_layout()

        ## MDeffect and MDCueOff
        #MSE,_,_ = self.do_test(20,self.MDeffect,True,False,
        #                        self.get_cue_list(),None,2)

        #return np.mean(MSE)

    def taskSwitch2(self,Nblock):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
        task1Cues = self.get_cue_list(0)
        task2Cues = self.get_cue_list(1)
        self.do_test(Nblock,self.MDeffect,True,False,
                    task1Cues,task1Cues[0],0,train=True)
        self.do_test(Nblock,self.MDeffect,False,False,
                    task2Cues,task2Cues[0],1,train=True)
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('fig_plasticPFC2Out.png',
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

    def taskSwitch3(self,Nblock,MDoff=True):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
        task1Cues = self.get_cue_list(0)
        task2Cues = self.get_cue_list(1)
        # after learning, during testing the learning rate is low, just performance tuning
        self.learning_rate /= 100.
        MSEs1,_,wOuts1 = self.do_test(Nblock,self.MDeffect,False,False,\
                            task1Cues,task1Cues[0],0,train=True)
        if MDoff:
            self.learning_rate *= 100.
            MSEs2,_,wOuts2 = self.do_test(Nblock,self.MDeffect,MDoff,False,\
                                task2Cues,task2Cues[0],1,train=True)
            self.learning_rate /= 100.
        else:
            MSEs2,_,wOuts2 = self.do_test(Nblock,self.MDeffect,MDoff,False,\
                                task2Cues,task2Cues[0],1,train=True)
        MSEs3,_,wOuts3 = self.do_test(Nblock,self.MDeffect,False,False,\
                            task1Cues,task1Cues[0],2,train=True)
        self.learning_rate *= 100.
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('fig_plasticPFC2Out.png',
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

            # plot the evolution of mean squared errors over each block
            fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
            ax2 = fig2.add_subplot(111)
            ax2.plot(MSEs1,'-,r')
            #ax2.plot(MSEs2,'-,b')
            ax2.plot(MSEs3,'-,g')

            # plot the evolution of different sets of weights
            fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
            ax2 = fig2.add_subplot(231)
            ax2.plot(np.reshape(wOuts1[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(232)
            ax2.plot(np.reshape(wOuts2[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(233)
            ax2.plot(np.reshape(wOuts3[:,:,:self.Nsub*2],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(234)
            ax2.plot(np.reshape(wOuts1[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(235)
            ax2.plot(np.reshape(wOuts2[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))
            ax2 = fig2.add_subplot(236)
            ax2.plot(np.reshape(wOuts3[:,:,self.Nsub*2:self.Nsub*4],(Nblock,-1)))
            ax2.set_ylim((-0.1,0.1))

    def test(self,Ntest):
        if self.plotFigs:
            self.fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*1.5),
                                facecolor='w')
            self.fig2 = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth),
                                facecolor='w')
        cues = self.get_cue_list()
        
        # after learning, during testing the learning rate is low, just performance tuning
        self.learning_rate /= 100.
        
        self.do_test(Ntest,self.MDeffect,False,False,cues,(0,0),0)
        if self.plotFigs:
            ax = self.fig2.add_subplot(111)
            # plot mean activity of each neuron for this taski+cuei
            #  further binning 10 neurons into 1
            ax.plot(np.mean(np.reshape(\
                                np.mean(self.meanAct[0,:,:],axis=0),\
                            (self.Nneur//10,10)),axis=1),',-r')
        if self.saveData:
            self.fileDict['meanAct0'] = self.meanAct[0,:,:]
        self.do_test(Ntest,self.MDeffect,False,False,cues,(0,1),1)
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
            self.do_test(Ntest,self.MDeffect,True,False,cues,(0,2),2)
            self.do_test(Ntest,self.MDeffect,True,False,cues,(0,3),3)
        else:
            self.do_test(Ntest,self.MDeffect,True,False,cues,(1,0),2)
            self.do_test(Ntest,self.MDeffect,True,False,cues,(1,1),3)
            #self.learning_rate *= 100
            ## MDeffect and MDCueOff
            #self.do_test(Ntest,self.MDeffect,True,False,cues,self.cuePlot,2)
            ## MDeffect and MDDelayOff
            ## network doesn't (shouldn't) learn this by construction.
            #self.do_test(Ntest,self.MDeffect,False,True,cues,self.cuePlot,3)
            ## back to old learning rate
            #self.learning_rate *= 100.
        
        if self.plotFigs:
            self.fig.tight_layout()
            self.fig.savefig('fig_plasticPFC2Out.png',
                        dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
            self.fig2.tight_layout()

    def load(self,filename):
        d = shelve.open(filename) # open
        if self.outExternal:
            self.wOut = d['wOut']
        else:
            self.Jrec[-self.Nout:,:] = d['JrecOut']
        if self.dirConn:
            self.wDir = d['wDir']
        d.close()
        return None

    def save(self):
        if self.outExternal:
            self.fileDict['wOut'] = self.wOut
        else:
            self.fileDict['JrecOut'] = self.Jrec[-self.Nout:,:]
        if self.dirConn:
            self.fileDict['wDir'] = self.wDir

if __name__ == "__main__":
    #PFC_G = 1.6                    # if not positiveRates
    PFC_G = 6.
    PFC_G_off = 1.5
    learning_rate = 5e-6
    learning_cycles_per_task = 1000#1000
    Ntest = 20
    Nblock = 70
    noiseSD = 1e-3
    tauError = 0.001
    reLoadWeights = False
    saveData = not reLoadWeights
    plotFigs = True#not saveData
    pfcmd = PFCMD(PFC_G,PFC_G_off,learning_rate,
                    noiseSD,tauError,plotFigs=plotFigs,saveData=saveData)
    if not reLoadWeights:
        pfcmd.train(learning_cycles_per_task)
        pfcmd.test_new(500)
        #pfcmd.train(learning_cycles_per_task)
#        if saveData:
#            pfcmd.save()
#        # save weights right after training,
#        #  since test() keeps training on during MD off etc.
#        pfcmd.test(Ntest)
    else:
        pfcmd.load(filename)
        # all 4cues in a block
        pfcmd.test(Ntest)
        
        #pfcmd.taskSwitch2(Nblock)
        
        # task switch
        #pfcmd.taskSwitch3(Nblock,MDoff=True)
        
        # control experiment: task switch without turning MD off
        # also has 2 cues in a block, instead of 4 as in test()
        #pfcmd.taskSwitch3(Nblock,MDoff=False)
    
#    if pfcmd.saveData:
#        pfcmd.fileDict.close()
    
    plt.show()
