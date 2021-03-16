import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import sys, shelve
import plot_utils as pltu
import pickle
from scipy.special import softmax
from sklearn.preprocessing import minmax_scale


def plot_column(self, fig, cues, routs, MDouts, outs, ploti=0):
    print('Plotting ...')
    cols = 4
    if ploti == 0:
        yticks = (0, 1)
        ylabels = ('Cues', 'PFC for cueA', 'PFC for cueB',
                   'PFC for cueC', 'PFC for cueD', 'PFC for rest',
                   'MD 1,2', 'Output 1,2')
    else:
        yticks = ()
        ylabels = ('', '', '', '', '', '', '', '')
    ax = fig.add_subplot(8, cols, 1 + ploti)
    ax.plot(cues, linewidth=pltu.plot_linewidth)
    ax.set_ylim([-0.1, 1.1])
    pltu.beautify_plot(ax, x0min=False, y0min=False,
                       xticks=(), yticks=yticks)
    pltu.axes_labels(ax, '', ylabels[0])
    ax = fig.add_subplot(8, cols, cols + 1 + ploti)
    ax.plot(routs[:, :10], linewidth=pltu.plot_linewidth)
    ax.set_ylim([-0.1, 1.1])
    pltu.beautify_plot(ax, x0min=False, y0min=False,
                       xticks=(), yticks=yticks)
    pltu.axes_labels(ax, '', ylabels[1])
    ax = fig.add_subplot(8, cols, cols * 2 + 1 + ploti)
    ax.plot(routs[:, self.Nsub:self.Nsub + 10],
            linewidth=pltu.plot_linewidth)
    ax.set_ylim([-0.1, 1.1])
    pltu.beautify_plot(ax, x0min=False, y0min=False,
                       xticks=(), yticks=yticks)
    pltu.axes_labels(ax, '', ylabels[2])
    if self.Ncues > 2:
        ax = fig.add_subplot(8, cols, cols * 3 + 1 + ploti)
        ax.plot(routs[:, self.Nsub * 2:self.Nsub * 2 + 10],
                linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1, 1.1])
        pltu.beautify_plot(ax, x0min=False, y0min=False,
                           xticks=(), yticks=yticks)
        pltu.axes_labels(ax, '', ylabels[3])
        ax = fig.add_subplot(8, cols, cols * 4 + 1 + ploti)
        ax.plot(routs[:, self.Nsub * 3:self.Nsub * 3 + 10],
                linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1, 1.1])
        pltu.beautify_plot(ax, x0min=False, y0min=False,
                           xticks=(), yticks=yticks)
        pltu.axes_labels(ax, '', ylabels[4])
        ax = fig.add_subplot(8, cols, cols * 5 + 1 + ploti)
        ax.plot(routs[:, self.Nsub * 4:self.Nsub * 4 + 10],
                linewidth=pltu.plot_linewidth)
        ax.set_ylim([-0.1, 1.1])
        pltu.beautify_plot(ax, x0min=False, y0min=False,
                           xticks=(), yticks=yticks)
        pltu.axes_labels(ax, '', ylabels[5])
    ax = fig.add_subplot(8, cols, cols * 6 + 1 + ploti)
    ax.plot(MDouts, linewidth=pltu.plot_linewidth)
    ax.set_ylim([-0.1, 1.1])
    pltu.beautify_plot(ax, x0min=False, y0min=False,
                       xticks=(), yticks=yticks)
    pltu.axes_labels(ax, '', ylabels[6])
    ax = fig.add_subplot(8, cols, cols * 7 + 1 + ploti)
    ax.plot(outs, linewidth=pltu.plot_linewidth)
    ax.set_ylim([-0.1, 1.1])
    pltu.beautify_plot(ax, x0min=False, y0min=False,
                       xticks=[0, self.tsteps], yticks=yticks)
    pltu.axes_labels(ax, 'time (ms)', ylabels[7])
    fig.tight_layout()

    if self.saveData:
        d = {}
        # 1st column of all matrices is number of time steps
        # 2nd column is number of neurons / units
        d['cues'] = cues  # tsteps x 4
        d['routs'] = routs  # tsteps x 1000
        d['MDouts'] = MDouts  # tsteps x 2
        d['outs'] = outs  # tsteps x 2
        savemat('simData' + str(ploti), d)

    return ax


def performance(self, cuei, outs, errors, target):
    meanErr = np.mean(errors[-100:, :] * errors[-100:, :])
    # endout is the mean of all end 100 time points for each output
    endout = np.mean(outs[-100:, :], axis=0)
    targeti = 0 if target[0] > target[1] else 1
    non_targeti = 1 if target[0] > target[1] else 0
    ## endout for targeti output must be greater than for the other
    ##  with a margin of 50% of desired difference of 1. between the two
    # if endout[targeti] > (endout[non_targeti]+0.5): correct = 1
    # else: correct = 0
    # just store the margin of error instead of thresholding it
    correct = endout[targeti] - endout[non_targeti]
    return meanErr, correct


def test_new(self, Ntest):
    MDeffect = self.MDeffect
    Ntest *= self.Ntasks
    wOuts = np.zeros(shape=(Ntest, self.Nout, self.Nneur))

    cues_all = np.zeros(shape=(Ntest, self.tsteps, self.Ncues))
    routs_all = np.zeros(shape=(Ntest, self.tsteps, self.Nneur))
    MDouts_all = np.zeros(shape=(Ntest, self.tsteps, self.Num_MD))
    MDinps_all = np.zeros(shape=(Ntest, self.tsteps, self.Num_MD))
    outs_all = np.zeros(shape=(Ntest, self.tsteps, self.Nout))
    MDoutTraces_all = np.zeros(shape=(Ntest, self.tsteps, self.Num_MD))

    MSEs = np.zeros(Ntest)
    for testi in range(Ntest):
        print('Simulating test cycle', testi)
        cueList = self.get_cue_list()
        cues_order = self.get_cues_order(cueList)
        for taski, cuei in cues_order:
            cue, target = \
                self.get_cue_target(taski, cuei)
            cues, routs, outs, MDouts, errors, MDinps, MDoutTraces, preTraces, postTraces = \
                self.sim_cue(taski, cuei, cue, target, MDeffect=MDeffect,
                             train=True)

            MSEs[testi] += np.mean(errors * errors)

            wOuts[testi, :, :] = self.wOut

            cues_all[testi, :, :] = cues
            routs_all[testi, :, :] = routs
            MDouts_all[testi, :, :] = MDouts
            MDinps_all[testi, :, :] = MDinps
            outs_all[testi, :, :] = outs
            MDoutTraces_all[testi, :, :] = MDoutTraces

    self.meanAct /= Ntest

    if self.saveData:
        #            self.fileDict['MSEs'] = MSEs
        #            self.fileDict['wOuts'] = wOuts

        pickle_out = open('dataPFCMD/test_HebbPostTrace_numMD' + str(
            self.Num_MD) + '_numTask' + str(self.Ntasks) + '_MD' + \
                          str(self.MDeffect) + \
                          '_Learn' + str(self.MDlearn) + \
                          '_R' + str(self.RNGSEED) + \
                          '_TimesTau' + str(self.tau_times) + \
                          '_Noise' + \
                          '.pickle', 'wb')
        pickle.dump(
            {'MSEs': MSEs, 'cues_all': cues_all, 'routs_all': routs_all, \
             'MDouts_all': MDouts_all, 'MDinps_all': MDinps_all,
             'outs_all': outs_all, 'MDoutTraces_all': MDoutTraces_all},
            pickle_out)
        pickle_out.close()


def do_test(self, Ntest, MDeffect, MDCueOff, MDDelayOff,
            cueList, cuePlot, colNum, train=True):
    NcuesTest = len(cueList)
    MSEs = np.zeros(Ntest * NcuesTest)
    corrects = np.zeros(Ntest * NcuesTest)
    wOuts = np.zeros((Ntest, self.Nout, self.Nneur))
    self.meanAct = np.zeros(shape=(self.Ntasks * self.inpsPerTask, \
                                   self.tsteps, self.Nneur))
    for testi in range(Ntest):
        if self.plotFigs: print('Simulating test cycle', testi)
        cues_order = self.get_cues_order(cueList)
        for cuenum, (taski, cuei) in enumerate(cues_order):
            cue, target = self.get_cue_target(taski, cuei)
            cues, routs, outs, MDouts, errors, MDinps = \
                self.sim_cue(taski, cuei, cue, target,
                             MDeffect, MDCueOff, MDDelayOff, train=train)
            MSEs[testi * NcuesTest + cuenum], corrects[
                testi * NcuesTest + cuenum] = \
                self.performance(cuei, outs, errors, target)

            if cuePlot is not None:
                if self.plotFigs and testi == 0 and taski == cuePlot[
                    0] and cuei == cuePlot[1]:
                    ax = self.plot_column(self.fig, cues, routs, MDouts,
                                          outs, ploti=colNum)

        if self.outExternal:
            wOuts[testi, :, :] = self.wOut

    self.meanAct /= Ntest
    if self.plotFigs and cuePlot is not None:
        ax.text(0.1, 0.4, '{:1.2f}$\pm${:1.2f}'.format(np.mean(corrects),
                                                       np.std(corrects)),
                transform=ax.transAxes)
        ax.text(0.1, 0.25,
                '{:1.2f}$\pm${:1.2f}'.format(np.mean(MSEs), np.std(MSEs)),
                transform=ax.transAxes)

    if self.saveData:
        # 1-Dim: numCycles * 4 cues/cycle i.e. 70*4=280
        self.fileDict['corrects' + str(colNum)] = corrects
        # at each cycle, a weights matrix 2x1000:
        # weights to 2 output neurons from 1000 cue-selective neurons
        # 3-Dim: 70 (numCycles) x 2 x 1000
        self.fileDict['wOuts' + str(colNum)] = wOuts
        # savemat('simDataTrials'+str(colNum), d)

    return MSEs, corrects, wOuts


def get_cue_list(self, taski=None):
    if taski is not None:
        # (taski,cuei) combinations for one given taski
        cueList = np.dstack((np.repeat(taski, self.inpsPerTask),
                             np.arange(self.inpsPerTask)))
    else:
        # every possible (taski,cuei) combination
        cueList = np.dstack(
            (np.repeat(np.arange(self.Ntasks), self.inpsPerTask),
             np.tile(np.arange(self.inpsPerTask), self.Ntasks)))
    return cueList[0]


def train(self, Ntrain):
    MDeffect = self.MDeffect
    if self.blockTrain:
        Nextra = 200  # add cycles to show if block1 learning is remembered
        Ntrain = Ntrain * self.Ntasks + Nextra
    else:
        Ntrain *= self.Ntasks
    wOuts = np.zeros(shape=(Ntrain * 2, self.Nout, self.Nneur))

    # Reset the trained weights,
    #  earlier for iterating over MDeffect = False and then True
    if self.outExternal:
        self.wOut = np.random.uniform(-1, 1,
                                      size=(
                                          self.Nout, self.Nneur)) / self.Nneur
        self.wOut *= self.wOutMask
    elif not MDeffect:
        self.Jrec[-self.Nout:, :] = \
            np.random.normal(size=(self.Nneur, self.Nneur)) \
            * self.G / np.sqrt(self.Nsub * 2)
    # direct connections from cue to output,
    #  similar to having output neurons within reservoir
    if self.dirConn:
        self.wDir = np.random.uniform(-1, 1,
                                      size=(self.Nout, self.Ncues)) \
                    / self.Ncues * 1.5

    MSEs = np.zeros(Ntrain)
    for traini in range(Ntrain):
        if self.plotFigs: print('Simulating training cycle', traini)

            cues, routs, outs, MDouts, errors, MDinps, MDoutTraces, preTraces, postTraces = \
                self.sim_cue(taski, cuei, cue, target, MDeffect=MDeffect,
                             train=True)
    self.meanAct /= Ntrain

    if self.plotFigs:
        self.fig2 = plt.figure(
            figsize=(pltu.columnwidth, pltu.columnwidth),
            facecolor='w')
        ax2 = self.fig2.add_subplot(1, 1, 1)
        ax2.plot(MSEs)
        pltu.beautify_plot(ax2, x0min=False, y0min=False)
        pltu.axes_labels(ax2, 'cycle num', 'MSE')
        self.fig2.tight_layout()

        # plot output weights evolution
        self.fig3 = plt.figure(
            figsize=(pltu.columnwidth, pltu.columnwidth),
            facecolor='w')
        ax3 = self.fig3.add_subplot(2, 1, 1)
        ax3.plot(wOuts[0::2, 0, :5], '-,r')
        ax3.plot(wOuts[0::2, 1, :5], '-,b')
        pltu.beautify_plot(ax3, x0min=False, y0min=False)
        pltu.axes_labels(ax3, 'cycle num', 'wAto0(r) wAto1(b)')
        ax4 = self.fig3.add_subplot(2, 1, 2)
        ax4.plot(wOuts[0::2, 0, self.Nsub:self.Nsub + 5], '-,r')
        ax4.plot(wOuts[0::2, 1, self.Nsub:self.Nsub + 5], '-,b')
        pltu.beautify_plot(ax4, x0min=False, y0min=False)
        pltu.axes_labels(ax4, 'cycle num', 'wBto0(r) wBto1(b)')
        self.fig3.tight_layout()

        if self.MDlearn:
            # plot PFC2MD weights evolution
            self.fig3 = plt.figure(
                figsize=(pltu.columnwidth, pltu.columnwidth),
                facecolor='w')
            ax3 = self.fig3.add_subplot(2, 1, 1)
            ax3.plot(wPFC2MDs[0::2, 0, :5], '-,r')
            ax3.plot(wPFC2MDs[0::2, 0, self.Nsub * 2:self.Nsub * 2 + 5],
                     '-,b')
            pltu.beautify_plot(ax3, x0min=False, y0min=False)
            pltu.axes_labels(ax3, 'cycle num', 'wAtoMD0(r) wCtoMD0(b)')
            ax4 = self.fig3.add_subplot(2, 1, 2)
            ax4.plot(wPFC2MDs[0::2, 1, :5], '-,r')
            ax4.plot(wPFC2MDs[0::2, 1, self.Nsub * 2:self.Nsub * 2 + 5],
                     '-,b')
            pltu.beautify_plot(ax4, x0min=False, y0min=False)
            pltu.axes_labels(ax4, 'cycle num', 'wAtoMD1(r) wCtoMD1(b)')
            self.fig3.tight_layout()

            # plot MD2PFC weights evolution
            self.fig3 = plt.figure(
                figsize=(pltu.columnwidth, pltu.columnwidth),
                facecolor='w')
            ax3 = self.fig3.add_subplot(2, 1, 1)
            ax3.plot(wMD2PFCs[0::2, :5, 0], '-,r')
            ax3.plot(wMD2PFCs[0::2, self.Nsub * 2:self.Nsub * 2 + 5, 0],
                     '-,b')
            pltu.beautify_plot(ax3, x0min=False, y0min=False)
            pltu.axes_labels(ax3, 'cycle num', 'wMD0toA(r) wMD0toC(b)')
            ax4 = self.fig3.add_subplot(2, 1, 2)
            ax4.plot(wMD2PFCMults[0::2, :5, 0], '-,r')
            ax4.plot(
                wMD2PFCMults[0::2, self.Nsub * 2:self.Nsub * 2 + 5, 0],
                '-,b')
            pltu.beautify_plot(ax4, x0min=False, y0min=False)
            pltu.axes_labels(ax4, 'cycle num', 'MwMD0toA(r) MwMD0toC(b)')
            self.fig3.tight_layout()

            # plot PFC to MD pre Traces
            self.fig3 = plt.figure(
                figsize=(pltu.columnwidth, pltu.columnwidth),
                facecolor='w')
            ax3 = self.fig3.add_subplot(1, 1, 1)
            ax3.plot(MDpreTraces[0::2, :5], '-,r')
            ax3.plot(MDpreTraces[0::2, self.Nsub * 2:self.Nsub * 2 + 5],
                     '-,b')
            pltu.beautify_plot(ax3, x0min=False, y0min=False)
            pltu.axes_labels(ax3, 'cycle num', 'cueApre(r) cueCpre(b)')
            self.fig3.tight_layout()

    ## MDeffect and MDCueOff
    # MSE,_,_ = self.do_test(20,self.MDeffect,True,False,
    #                        self.get_cue_list(),None,2)

    # return np.mean(MSE)


def taskSwitch2(self, Nblock):
    if self.plotFigs:
        self.fig = plt.figure(
            figsize=(pltu.twocolumnwidth, pltu.twocolumnwidth * 1.5),
            facecolor='w')
    task1Cues = self.get_cue_list(0)
    task2Cues = self.get_cue_list(1)
    self.do_test(Nblock, self.MDeffect, True, False,
                 task1Cues, task1Cues[0], 0, train=True)
    self.do_test(Nblock, self.MDeffect, False, False,
                 task2Cues, task2Cues[0], 1, train=True)

    if self.plotFigs:
        self.fig.tight_layout()
        self.fig.savefig('fig_plasticPFC2Out.png',
                         dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')


def taskSwitch3(self, Nblock, MDoff=True):
    if self.plotFigs:
        self.fig = plt.figure(
            figsize=(pltu.twocolumnwidth, pltu.twocolumnwidth * 1.5),
            facecolor='w')
    task1Cues = self.get_cue_list(0)
    task2Cues = self.get_cue_list(1)
    # after learning, during testing the learning rate is low, just performance tuning
    self.learning_rate /= 100.
    MSEs1, _, wOuts1 = self.do_test(Nblock, self.MDeffect, False, False, \
                                    task1Cues, task1Cues[0], 0, train=True)
    if MDoff:
        self.learning_rate *= 100.
        MSEs2, _, wOuts2 = self.do_test(Nblock, self.MDeffect, MDoff,
                                        False, \
                                        task2Cues, task2Cues[0], 1,
                                        train=True)
        self.learning_rate /= 100.
    else:
        MSEs2, _, wOuts2 = self.do_test(Nblock, self.MDeffect, MDoff,
                                        False, \
                                        task2Cues, task2Cues[0], 1,
                                        train=True)
    MSEs3, _, wOuts3 = self.do_test(Nblock, self.MDeffect, False, False, \
                                    task1Cues, task1Cues[0], 2, train=True)
    self.learning_rate *= 100.

    if self.plotFigs:
        self.fig.tight_layout()
        self.fig.savefig('fig_plasticPFC2Out.png',
                         dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')

        # plot the evolution of mean squared errors over each block
        fig2 = plt.figure(
            figsize=(pltu.twocolumnwidth, pltu.twocolumnwidth),
            facecolor='w')
        ax2 = fig2.add_subplot(111)
        ax2.plot(MSEs1, '-,r')
        # ax2.plot(MSEs2,'-,b')
        ax2.plot(MSEs3, '-,g')

        # plot the evolution of different sets of weights
        fig2 = plt.figure(
            figsize=(pltu.twocolumnwidth, pltu.twocolumnwidth),
            facecolor='w')
        ax2 = fig2.add_subplot(231)
        ax2.plot(np.reshape(wOuts1[:, :, :self.Nsub * 2], (Nblock, -1)))
        ax2.set_ylim((-0.1, 0.1))
        ax2 = fig2.add_subplot(232)
        ax2.plot(np.reshape(wOuts2[:, :, :self.Nsub * 2], (Nblock, -1)))
        ax2.set_ylim((-0.1, 0.1))
        ax2 = fig2.add_subplot(233)
        ax2.plot(np.reshape(wOuts3[:, :, :self.Nsub * 2], (Nblock, -1)))
        ax2.set_ylim((-0.1, 0.1))
        ax2 = fig2.add_subplot(234)
        ax2.plot(np.reshape(wOuts1[:, :, self.Nsub * 2:self.Nsub * 4],
                            (Nblock, -1)))
        ax2.set_ylim((-0.1, 0.1))
        ax2 = fig2.add_subplot(235)
        ax2.plot(np.reshape(wOuts2[:, :, self.Nsub * 2:self.Nsub * 4],
                            (Nblock, -1)))
        ax2.set_ylim((-0.1, 0.1))
        ax2 = fig2.add_subplot(236)
        ax2.plot(np.reshape(wOuts3[:, :, self.Nsub * 2:self.Nsub * 4],
                            (Nblock, -1)))
        ax2.set_ylim((-0.1, 0.1))


def test(self, Ntest):
    if self.plotFigs:
        self.fig = plt.figure(
            figsize=(pltu.twocolumnwidth, pltu.twocolumnwidth * 1.5),
            facecolor='w')
        self.fig2 = plt.figure(
            figsize=(pltu.twocolumnwidth, pltu.twocolumnwidth),
            facecolor='w')
    cues = self.get_cue_list()

    # after learning, during testing the learning rate is low, just performance tuning
    self.learning_rate /= 100.

    self.do_test(Ntest, self.MDeffect, False, False, cues, (0, 0), 0)
    if self.plotFigs:
        ax = self.fig2.add_subplot(111)
        # plot mean activity of each neuron for this taski+cuei
        #  further binning 10 neurons into 1
        ax.plot(np.mean(np.reshape( \
            np.mean(self.meanAct[0, :, :], axis=0), \
            (self.Nneur // 10, 10)), axis=1), ',-r')
    if self.saveData:
        self.fileDict['meanAct0'] = self.meanAct[0, :, :]
    self.do_test(Ntest, self.MDeffect, False, False, cues, (0, 1), 1)
    if self.plotFigs:
        # plot mean activity of each neuron for this taski+cuei
        ax.plot(np.mean(np.reshape( \
            np.mean(self.meanAct[1, :, :], axis=0), \
            (self.Nneur // 10, 10)), axis=1), ',-b')
        ax.set_xlabel('neuron #')
        ax.set_ylabel('mean rate')
    if self.saveData:
        self.fileDict['meanAct1'] = self.meanAct[1, :, :]

    if self.xorTask:
        self.do_test(Ntest, self.MDeffect, True, False, cues, (0, 2), 2)
        self.do_test(Ntest, self.MDeffect, True, False, cues, (0, 3), 3)
    else:
        self.do_test(Ntest, self.MDeffect, True, False, cues, (1, 0), 2)
        self.do_test(Ntest, self.MDeffect, True, False, cues, (1, 1), 3)
        # self.learning_rate *= 100
        ## MDeffect and MDCueOff
        # self.do_test(Ntest,self.MDeffect,True,False,cues,self.cuePlot,2)
        ## MDeffect and MDDelayOff
        ## network doesn't (shouldn't) learn this by construction.
        # self.do_test(Ntest,self.MDeffect,False,True,cues,self.cuePlot,3)
        ## back to old learning rate
        # self.learning_rate *= 100.

    if self.plotFigs:
        self.fig.tight_layout()
        self.fig.savefig('fig_plasticPFC2Out.png',
                         dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
        self.fig2.tight_layout()