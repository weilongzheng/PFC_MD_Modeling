# -*- coding: utf-8 -*-
# (c) Jun 2018 Aditya Gilra, EPFL., Uni-Bonn

import numpy as np
import matplotlib.pyplot as plt
import sys,shelve
import plot_utils as pltu

def plot_axes(axs,MDstr,xorStr):
    RNGs = ('0','1','2','3','4','5','6','7','8','9')
    #RNGs = ('0',)
    Ntrain = 1000
    Nsub = 200
    Nneur = 1000
    numRNG = len(RNGs)
    deltaW = np.zeros((numRNG,2,3))
    MSEs = np.zeros((numRNG,Ntrain*2+200))
    meanActs = np.zeros((numRNG,2,Nneur))
    meanActsBase = np.zeros((numRNG,2,Nneur))
    for idx,RNG in enumerate(RNGs):
        print('Reading data for RNG =',RNG)
        fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD'+\
                        str(MDstr)+\
                        '_R'+str(RNG)+\
                        xorStr+'.shelve')
        MSEs[idx,:] = fileDict['MSEs']
        for i,(startidx,endidx) in enumerate(((0,Ntrain-1),(Ntrain,Ntrain*2-1),
                                            (Ntrain*2,Ntrain*2+200-1))):
            diffW = np.diff(fileDict['wOuts'],axis=0)
            for taski in range(2):
                deltaW[idx,taski,i] = np.mean(np.abs(np.mean(
                        diffW[startidx:endidx,:,taski*Nsub*2:(taski+1)*Nsub*2]
                                        ,axis=0))) * 1e5
            # obsolete
            #endWminusstartW = ( fileDict['wOuts'][endidx,:,:] - \
            #                    fileDict['wOuts'][startidx,:,:] )
            #endWplusstartW = ( fileDict['wOuts'][endidx,:,:] + \
            #                    fileDict['wOuts'][startidx,:,:] )
            ## return zero if denominator is zero:
            ##  https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
            #for taski in range(2):
            #    deltaW[idx,taski,i] = np.mean(np.abs( 100.* \
            #        np.divide(endWminusstartW[:,taski*Nsub*2:(taski+1)*Nsub*2],
            #            endWplusstartW[:,taski*Nsub*2:(taski+1)*Nsub*2]/2.,\
            #            out=np.zeros_like(endWminusstartW[:,taski*Nsub*2:(taski+1)*Nsub*2]),\
            #            where=endWplusstartW[:,taski*Nsub*2:(taski+1)*Nsub*2]!=0.) ))

            # obsolete
            #axs[1].plot(fileDict['wOuts'][:,0,100:105],'-,r')
            #axs[1].plot(fileDict['wOuts'][:,1,300:305],'-,b')
            
        meanActs[idx,0,:] = np.mean(fileDict['meanAct0'],axis=0)
        meanActs[idx,1,:] = np.mean(fileDict['meanAct1'],axis=0)
        fileDict.close()
        #fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD'+\
        #                '-1.0_R'+str(RNG)+'.shelve')
        #meanActsBase[idx,0,:] = np.mean(fileDict['meanAct0'],axis=0)
        #meanActsBase[idx,1,:] = np.mean(fileDict['meanAct1'],axis=0)
        #fileDict.close()

    if xorStr=='':
        Ncuespercycle = 2
        ylimMSE,ylimDW = 1,2
        xTrials,Nneurons = 4500,1000
    else:
        Ncuespercycle = 4
        ylimMSE,ylimDW = 2,3
        xTrials,Nneurons = 9000,1000
    trange = np.arange(MSEs.shape[1])*Ncuespercycle
    meanMSE = np.mean(MSEs,axis=0)
    stdMSE = np.std(MSEs,axis=0)
    axs[0].fill_between(trange,meanMSE-stdMSE,meanMSE+stdMSE,\
                            color='#777777')
    axs[0].plot(trange,meanMSE,color='k')
    axs[0].set_ylim([0,ylimMSE])
    axs[0].set_xlim([0,xTrials])
    pltu.beautify_plot(axs[0])
    pltu.axes_labels(axs[0],'Trial number','Mean Squared Error (MSE)',xpad=-2,ypad=-5)

    binW = 10
    #binW = 1
    def bin10(act):
       return np.mean(np.reshape(act,(Nneur//binW,binW)),axis=1)
       
    neurx = np.arange(0,Nneur,binW)
    colorList = ('b','r')
    colorList2 = ('#8080ff','#ff8080')
    #axs[1].plot(neurx,bin10(np.mean(meanActsBase[:,0,:],axis=0)),',-m')
    #axs[1].plot(neurx,bin10(np.mean(meanActsBase[:,1,:],axis=0)),',-c')
    #axs[1].set_ylim([0,1])
    #pltu.beautify_plot(axs[1])
    #pltu.axes_labels(axs[1],'neuron #','mean activity (arb)')
    # plot mean activity for each neuron in context 1 for cues 0,1
    for i in (0,1):
        meanAct = bin10(np.mean(meanActs[:,i,:],axis=0))
        stdAct = bin10(np.std(meanActs[:,i,:],axis=0))
        axs[1].fill_between(neurx,meanAct-stdAct,meanAct+stdAct,\
                        color=colorList2[i])
        axs[1].plot(neurx,meanAct,color=colorList[i])
    # plot thick coloured lines to highlight cue neurons
    for i in (1,0):
        cueAct = np.ones(Nsub//binW)*0.05
        axs[1].plot(neurx[Nsub*i//binW:Nsub*(i+1)//binW],\
                        cueAct,color=colorList[i],linewidth=3)
    axs[1].set_ylim([0,1])
    axs[1].set_xlim([0,Nneurons])
    pltu.beautify_plot(axs[1])
    pltu.axes_labels(axs[1],'Neuron number','Mean activity (au)',\
                        xpad=-2,ypad=-5)

    width = 0.5
    colorList = (('k','r'),('r','k'))
    for i in (0,1):
        # plot change in weight per trial for output weights of two contexts,
        #  when each context is being learned
        #axs[2].bar((width+width*i,), np.mean(deltaW[:,0,i],axis=0), width,
        #        yerr=np.std(deltaW[:,0,i],axis=0),
        #        align='center', color=colorList[i],
        #        edgecolor=colorList[i], ecolor=colorList[i], capsize=5)
        #axs[2].bar((width*5+width*i,), np.mean(deltaW[:,1,i],axis=0), width,
        #        yerr=np.std(deltaW[:,1,i],axis=0),
        #        align='center', color=colorList[i],
        #        edgecolor=colorList[i], ecolor=colorList[i], capsize=5)
        axs[2].boxplot(deltaW[:,0,i],positions=(width+width*i,),
                        widths=(width,),showcaps=False,whis='range',
                        boxprops=dict(color=colorList[i][0]),
                        whiskerprops=dict(color=colorList[i][0]),
                        medianprops=dict(color=colorList[i][0]))
        axs[2].boxplot(deltaW[:,1,i],positions=(width*5+width*i,),
                        widths=(width,),showcaps=False,whis='range',
                        boxprops=dict(color=colorList[i][1]),
                        whiskerprops=dict(color=colorList[i][1]),
                        medianprops=dict(color=colorList[i][1]))
    axs[2].set_ylim([0,ylimDW])
    pltu.beautify_plot(axs[2],xticks=(width*1.5,width*5.5))
    axs[2].set_xticklabels(('1','2'))
    # Note: arb unit for weights below is actually arb x 10^5
    pltu.axes_labels(axs[2],'Current context','$\Delta w$/trial (au)',\
                        xpad=-2,ypad=-5)
    
if __name__ == "__main__":
    # choose one of the below
    #xorStr = '_xor'
    xorStr = ''
    fig = plt.figure(figsize=(pltu.twocolumnwidth,pltu.twocolumnwidth*0.75),
                        facecolor='w')
    for idx,MDstr in enumerate(('0.0','1.0')):
        ax1 = fig.add_subplot(2,3,idx*3+1)
        ax2 = fig.add_subplot(2,3,idx*3+2)
        ax3 = fig.add_subplot(2,3,idx*3+3)
        plot_axes((ax1,ax2,ax3),MDstr,xorStr)
    fig.tight_layout()
    fig.savefig('fig_paper'+xorStr+'.eps', format='eps',
                dpi=pltu.fig_dpi, facecolor='w', edgecolor='w')
