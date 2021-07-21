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
    plt.plot(c.T,color='tab:red',linewidth=0.5)
    plt.xlabel('Neuron #')
    plt.ylabel('Mean Activity (a.u.)')
    plt.title('PFC (MD Off)')
    plt.ylim(0,0.9)
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_activity_noMD.pdf') 
    c=np.mean(routs_mean[cue2,:],axis=1)
    plt.figure(figsize=(2.4,2.4))
    plt.plot(c.T,color='tab:red',linewidth=0.5)
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
    norm_nomd = 0
    cue1=np.where(a[:,0]==1)
    import pdb;pdb.set_trace()  
    norm_nomd+=np.linalg.norm(np.mean(routs_mean[cue1,:],axis=1))
    cue2=np.where(a[:,1]==1)
    norm_nomd+=np.linalg.norm(np.mean(routs_mean[cue2,:],axis=1))
    cue3=np.where(a[:,2]==1)
    norm_nomd+=np.linalg.norm(np.mean(routs_mean[cue3,:],axis=1))
    cue4=np.where(a[:,3]==1)
    norm_nomd+=np.linalg.norm(np.mean(routs_mean[cue4,:],axis=1))
    norm_nomd=norm_nomd/4
    norm_nomd_std = np.std([np.linalg.norm(np.mean(routs_mean[cue1,:],axis=1)),np.linalg.norm(np.mean(routs_mean[cue2,:],axis=1)),np.linalg.norm(np.mean(routs_mean[cue3,:],axis=1)),np.linalg.norm(np.mean(routs_mean[cue4,:],axis=1))])
    
    file=open('files/final/test_numMD10_numContext2_MDTrue_R1.pkl','rb')
    data=pickle.load(file)
    cues_all=data['cues_all']
    a=cues_all[:,0,:]
    routs_all=data['PFCouts_all']
    routs_mean = np.mean(routs_all,axis=1)
    norm_md = 0
    cue1=np.where(a[:,0]==1)
    norm_md+=np.linalg.norm(np.mean(routs_mean[cue1,:],axis=1))
    cue2=np.where(a[:,1]==1)
    norm_md+=np.linalg.norm(np.mean(routs_mean[cue2,:],axis=1))
    cue3=np.where(a[:,2]==1)
    norm_md+=np.linalg.norm(np.mean(routs_mean[cue3,:],axis=1))
    cue4=np.where(a[:,3]==1)
    norm_md+=np.linalg.norm(np.mean(routs_mean[cue4,:],axis=1))
    norm_md=norm_md/4
    norm_md_std = np.std([np.linalg.norm(np.mean(routs_mean[cue1,:],axis=1)),np.linalg.norm(np.mean(routs_mean[cue2,:],axis=1)),np.linalg.norm(np.mean(routs_mean[cue3,:],axis=1)),np.linalg.norm(np.mean(routs_mean[cue4,:],axis=1))])
    
    
    bars=['Without MD','With MD']
    plt.figure(figsize=(1.8,2.4))
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, [norm_nomd,norm_md], yerr=[norm_nomd_std, norm_md_std], color=['tab:red', 'tab:blue'])
    plt.xticks(x_pos, bars)
    plt.ylabel('PFC Mean Activity Norm')
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_norm.pdf') 
    
if __name__ == '__main__':
    #plotDeltaW()
    plotRout()
    plotPFCnorm()