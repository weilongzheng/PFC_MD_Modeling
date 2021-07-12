# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 23:33:07 2021

@author: weilo
"""
mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
    # plot delta W 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib import ticker
FIGUREPATH = Path('./results')

def plotDeltaW():
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
    
if __name__ == '__main__':
    plotDeltaW()