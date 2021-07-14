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
    file = open('files/train_wout_numMD10_numContext2_MDFalse_PFCFalse_R10.pkl','rb')
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
    
    file = open('files/train_wout_numMD10_numContext2_MDTrue_PFCFalse_R10.pkl','rb')
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
    plt.plot(c.T)
    plt.xlabel('Neuron #')
    plt.ylabel('Mean Activity (a.u.)')
    plt.title('PFC (MD On)')
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_activity_MD.pdf') 
#    c=np.mean(routs_mean[cue2,:],axis=1)
#    plt.plot(c.T,color='r')
#    plt.xlabel('Neuron #')
#    plt.ylabel('Mean Activity')

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
    plt.plot(c.T)
    plt.xlabel('Neuron #')
    plt.ylabel('Mean Activity (a.u.)')
    plt.title('PFC (MD Off)')
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'pfc_activity_noMD.pdf') 
    
if __name__ == '__main__':
    #plotDeltaW()
    plotRout()