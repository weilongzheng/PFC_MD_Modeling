# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:33:07 2021

@author: weilo
"""
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib import ticker
from pathlib import Path
import pickle
import numpy as np

FIGUREPATH = Path('./results')
mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True

def plotDeltaW():
    '''
    # plot delta W 
    '''
    file = open('files/final/train_wout_numMD10_numContext2_MDFalse_PFCFalse_R10.pkl','rb')
    data = pickle.load(file)
    wOuts = data['Wout_all']
    
    a=np.diff(wOuts,axis=0)
    a = abs(a)
    deltaW = np.mean(np.mean(a[200:300,:,0:400],axis=0),axis=0)
    context = np.ones((400,))
    group = np.ones((400,))
    b = np.mean(np.mean(a[200:300,:,400:800],axis=0),axis=0)
    deltaW = np.append(deltaW,b)
    context = np.append(context,np.ones((400,)))
    group = np.append(group,0*np.ones((400,)))
    b = np.mean(np.mean(a[100:200,:,400:800],axis=0),axis=0)
    deltaW = np.append(deltaW,b)
    context = np.append(context,2*np.ones((400,)))
    group = np.append(group,np.ones((400,)))
    b = np.mean(np.mean(a[100:200,:,0:400],axis=0),axis=0)
    deltaW = np.append(deltaW,b)
    context = np.append(context,2*np.ones((400,)))
    group = np.append(group,0*np.ones((400,)))
    d = {'Delta W':deltaW,'Current-Context Neurons':group,'Context':context}
    df = pd.DataFrame(data=d)
    plt.figure(figsize=(2.4,2.4))
    ax = sns.boxplot(x='Context',y='Delta W',hue='Current-Context Neurons',data=df)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 
    ax.set_title('MD Off')
    matplotlib.pyplot.ylim(0,0.00055)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'deltaW_noMD.pdf') 
    
    file = open('files/final/train_wout_numMD10_numContext2_MDTrue_PFCFalse_R10.pkl','rb')
    data = pickle.load(file)
    wOuts = data['Wout_all']
    
    a=np.diff(wOuts,axis=0)
    a = abs(a)
    deltaW = np.mean(np.mean(a[200:300,:,0:400],axis=0),axis=0)
    context = np.ones((400,))
    group = np.ones((400,))
    b = np.mean(np.mean(a[200:300,:,400:800],axis=0),axis=0)
    deltaW = np.append(deltaW,b)
    context = np.append(context,np.ones((400,)))
    group = np.append(group,0*np.ones((400,)))
    b = np.mean(np.mean(a[100:200,:,400:800],axis=0),axis=0)
    deltaW = np.append(deltaW,b)
    context = np.append(context,2*np.ones((400,)))
    group = np.append(group,np.ones((400,)))
    b = np.mean(np.mean(a[100:200,:,0:400],axis=0),axis=0)
    deltaW = np.append(deltaW,b)
    context = np.append(context,2*np.ones((400,)))
    group = np.append(group,0*np.ones((400,)))
    d = {'Delta W':deltaW,'Current-Context Neurons':group,'Context':context}
    df = pd.DataFrame(data=d)
    plt.figure(figsize=(2.4,2.4))
    ax = sns.boxplot(x='Context',y='Delta W',hue='Current-Context Neurons',data=df)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 
    ax.set_title('MD On')
    matplotlib.pyplot.ylim(0,0.00055)
    plt.legend(frameon=False)
    plt.tight_layout()
    
def plotRout():
    '''
    plot pfc mean activity
    '''
    file = open('files/final/test_numMD10_numContext2_MDTrue_R1.pkl','rb')
    data = pickle.load(file)
    routs_all = data['PFCouts_all']
    cues_all = data['cues_all']
    a = cues_all[:,0,:]
    cue1 = np.where(a[:,0]==1)
    cue2 = np.where(a[:,1]==1)
    
    routs_mean = np.mean(routs_all,axis=1)
    c=np.mean(routs_mean[cue1,:],axis=1)
    plt.figure(figsize=(2.4,2.4))
    plt.plot(c.T,color='tab:blue',linewidth=1)
    plt.xlabel('Neuron #')
    plt.ylabel('Mean Activity (a.u.)')
    plt.title('PFC (MD On)')
    plt.ylim(0,0.9)
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_activity_MD.pdf') 
    c=np.mean(routs_mean[cue2,:],axis=1)
    plt.figure(figsize=(2.4,2.4))
    plt.plot(c.T,color='tab:blue',linewidth=1)
    plt.xlabel('Neuron #')
    plt.ylabel('Mean Activity')
    plt.title('PFC (MD On)')
    plt.ylim(0,0.9)
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_activity2_MD.pdf') 

    file = open('files/final/test_numMD10_numContext2_MDFalse_R1.pkl','rb')
    data = pickle.load(file)
    routs_all = data['PFCouts_all']
    cues_all = data['cues_all']
    a = cues_all[:,0,:]
    cue1 = np.where(a[:,0]==1)
    cue2 = np.where(a[:,1]==1)
    
    routs_mean = np.mean(routs_all,axis=1)
    c=np.mean(routs_mean[cue1,:],axis=1)
    plt.figure(figsize=(2.4,2.4))
    plt.plot(c.T,color='tab:orange',linewidth=1)
    plt.xlabel('Neuron #')
    plt.ylabel('Mean Activity (a.u.)')
    plt.title('PFC (MD Off)')
    plt.ylim(0,0.9)
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_activity_noMD.pdf') 
    c=np.mean(routs_mean[cue2,:],axis=1)
    plt.figure(figsize=(2.4,2.4))
    plt.plot(c.T,color='tab:orange',linewidth=1)
    plt.xlabel('Neuron #')
    plt.ylabel('Mean Activity')
    plt.title('PFC (MD Off)')
    plt.ylim(0,0.9)
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_activity2_noMD.pdf') 
    
    
def plotPFCnorm():
    file=open('files/final/test_numMD10_numContext2_MDFalse_R1.pkl','rb')
    data=pickle.load(file)
    cues_all=data['cues_all']
    a=cues_all[:,0,:]
    routs_all=data['PFCouts_all']
    routs_mean = np.mean(routs_all,axis=1)
    routs_mean_nomd = routs_mean;
    norm_nomd = 0
    cue1=np.where(a[:,0]==1)
    
    norm_nomd+=np.mean(np.linalg.norm(routs_mean[cue1,:],axis=2))
    cue2=np.where(a[:,1]==1)
    norm_nomd+=np.mean(np.linalg.norm(routs_mean[cue2,:],axis=2))
    cue3=np.where(a[:,2]==1)
    norm_nomd+=np.mean(np.linalg.norm(routs_mean[cue3,:],axis=2))
    cue4=np.where(a[:,3]==1)
    norm_nomd+=np.mean(np.linalg.norm(routs_mean[cue4,:],axis=2))
    norm_nomd=norm_nomd/4
    norm_nomd_std = np.std([np.linalg.norm(routs_mean[cue1,:],axis=2),np.linalg.norm(routs_mean[cue2,:],axis=2),np.linalg.norm(routs_mean[cue3,:],axis=2),np.linalg.norm(routs_mean[cue4,:],axis=2)])
    
    file=open('files/final/test_numMD10_numContext2_MDTrue_R1.pkl','rb')
    data=pickle.load(file)
    cues_all=data['cues_all']
    a=cues_all[:,0,:]
    routs_all=data['PFCouts_all']
    routs_mean = np.mean(routs_all,axis=1)
    routs_mean_md = routs_mean;
    norm_md = 0
    cue1=np.where(a[:,0]==1)
    norm_md+=np.mean(np.linalg.norm(routs_mean[cue1,:],axis=2))
    cue2=np.where(a[:,1]==1)
    norm_md+=np.mean(np.linalg.norm(routs_mean[cue2,:],axis=2))
    cue3=np.where(a[:,2]==1)
    norm_md+=np.mean(np.linalg.norm(routs_mean[cue3,:],axis=2))
    cue4=np.where(a[:,3]==1)
    norm_md+=np.mean(np.linalg.norm(routs_mean[cue4,:],axis=2))
    norm_md=norm_md/4
    norm_md_std = np.std([np.linalg.norm(routs_mean[cue1,:],axis=2),np.linalg.norm(routs_mean[cue2,:],axis=2),np.linalg.norm(routs_mean[cue3,:],axis=2),np.linalg.norm(routs_mean[cue4,:],axis=2)])
    
    bars=['Without MD','With MD']
    plt.figure(figsize=(1.8,2.4))
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, [norm_nomd,norm_md], yerr=[norm_nomd_std, norm_md_std], color=['tab:orange', 'tab:blue'])
    plt.xticks(x_pos, bars)
    plt.ylabel('PFC Mean Activity Norm')
    plt.title('PFC Activity Norm')
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_norm.pdf') 
    
    #import pdb;pdb.set_trace()  
    fig = plt.figure(figsize=(2.4,2.4))
    bins = np.arange(0,0.7,0.1)
    data_plot = np.array([np.mean(routs_mean_md,axis=0),np.mean(routs_mean_nomd,axis=0)])
    ax = sns.histplot(data=data_plot.T, stat = 'probability', bins = bins, kde=True, label={'Without MD','With MD'})
    ax.legend()
    ax.set(xlabel='Activity (a.u.)', ylabel='Probability', title='Distribution')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.show()
    plt.savefig(FIGUREPATH/'pfc_distribution.pdf') 

    wPFC2MD = log['wPFC2MD']
    wMD2PFC = log['wMD2PFC']
    ax = plt.figure()
    ax = sns.heatmap(wPFC2MD, cmap='Reds')
    ax.set_xticks([0, 999])
    ax.set_xticklabels([1, 1000], rotation=0)
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_xlabel('PFC neuron index')
    ax.set_ylabel('MD neuron index')
    ax.set_title('wPFC2MD '+'PFC learnable-'+str(PFClearn))
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight')
    plt.tight_layout()
    plt.show()

def plotWevolution():
    file = open('files/train_allVarT_numMD10_numContext3_MDTrue_PFCFalse_R10.pkl','rb')
    data = pickle.load(file)
    wPFC2MDs_all = data['wPFC2MDs_all']
    wMD2PFCs_all = data['wMD2PFCs_all']
    Ntrain = data['Ntrain']
    
    for i in np.arange(0,4):

        wPFC2MD = wPFC2MDs_all[(i+1)*Ntrain*2-1,199,:,:]
        wMD2PFC = wMD2PFCs_all[(i+1)*Ntrain*2-1,199,:,:]
        ax = plt.figure(figsize=(2.4,2))
        ax = sns.heatmap(wPFC2MD, cmap='Reds')
        ax.set_xticks([0, 399, 799, 1199, 1399])
        ax.set_xticklabels([1, 400, 800, 1200, 1400], rotation=0)
        ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
        ax.set_xlabel('PFC Index')
        ax.set_ylabel('MD Index')
        ax.set_title('wPFC2MD')
        cbar = ax.collections[0].colorbar
        plt.tight_layout()
        plt.show()
        file_name = 'wPFC2MD_t'+str(i+1)+'.pdf'
        plt.savefig(FIGUREPATH/file_name) 

        # Heatmap wMD2PFC
        ax = plt.figure(figsize=(2.4,2))
        ax = sns.heatmap(wMD2PFC, cmap='Blues_r')
        ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
        ax.set_yticks([0, 399, 799, 1199, 1399])
        ax.set_yticklabels([1, 400, 800, 1200, 1400], rotation=0)
        ax.set_xlabel('MD Index')
        ax.set_ylabel('PFC Index')
        ax.set_title('wMD2PFC')
        cbar = ax.collections[0].colorbar
        plt.tight_layout()
        plt.show()
        file_name = 'wMD2PFC_t'+str(i+1)+'.pdf'
        plt.savefig(FIGUREPATH/file_name) 

def plotMDpretraces():
    file = open('files/train_allVarT_numMD10_numContext2_MDTrue_PFCFalse_R10.pkl','rb')
    data = pickle.load(file)
    wPFC2MDs_all = data['wPFC2MDs_all']
    wMD2PFCs_all = data['wMD2PFCs_all']
    MDpreTraces_all = data['MDpreTraces_all']
    MDouts_all = data['MDouts_all']
    
    plot_t = 30 #10 30
    ax = plt.figure(figsize=(2.4,2))
    ax = sns.heatmap(MDpreTraces_all[plot_t,:,:].T, cmap='hot')
    ax.set_yticks([0, 399, 799, 999])
    ax.set_yticklabels([1, 400, 800, 1000], rotation=0)
    ax.set_xticks([0, 49, 99, 149, 199])
    ax.set_xticklabels([1, 50, 100, 150, 200], rotation=0)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('PFC Index')
    ax.set_title('MD Presynaptic Trace')
    plt.tight_layout()
    plt.show()
    file_name = 'MDpreTraces_t'+str(plot_t)+'.pdf'
    plt.savefig(FIGUREPATH/file_name) 
    
    ax = plt.figure(figsize=(2.4,2))
    ax = sns.heatmap(MDouts_all[plot_t,:,:].T, cmap='Blues')
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_xticks([0, 49, 99, 149, 199])
    ax.set_xticklabels([1, 50, 100, 150, 200], rotation=0)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('MD Index')
    ax.set_title('MD Outputs')
    plt.tight_layout()
    plt.show()
    file_name = 'MDouts_t'+str(plot_t)+'.pdf'
    plt.savefig(FIGUREPATH/file_name) 
    
    #import pdb;pdb.set_trace() 
    ax = plt.figure(figsize=(1.8,2))
    temp = wPFC2MDs_all[plot_t,:,:,:]
    wPFC2MDs = np.mean(temp[:,4,:400],axis=1)
    ax = plt.plot(wPFC2MDs,'g-',label='Cxt 1 to MD A')
    temp = wPFC2MDs_all[plot_t,:,:,:]
    wPFC2MDs = np.mean(temp[:,1,:400],axis=1)
    ax = plt.plot(wPFC2MDs,'g--',label='Cxt 1 to MD B')
    temp = wPFC2MDs_all[plot_t,:,:,:]
    wPFC2MDs = np.mean(temp[:,4,400:800],axis=1)
    ax = plt.plot(wPFC2MDs,'-',color='tab:orange',label='Cxt 2 to MD A')
    temp = wPFC2MDs_all[plot_t,:,:,:]
    wPFC2MDs = np.mean(temp[:,1,400:800],axis=1)
    ax = plt.plot(wPFC2MDs,'--',color='tab:orange',label='Cxt 2 to MD B')
    plt.xlabel('Time Steps')
    plt.ylabel('Mean Weights')
    plt.title('Weight Learning')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    file_name = 'Weights_t'+str(plot_t)+'.pdf'
    plt.savefig(FIGUREPATH/file_name) 
    
if __name__ == '__main__':
    #plotDeltaW()
    #plotRout()
    #plotPFCnorm()
    #plotWevolution()
    plotMDpretraces()