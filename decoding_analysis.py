# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:14:14 2020

@author: weilong
"""

"""Decoding analysis."""

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import matplotlib as mpl
import seaborn as sns
MODELPATH = Path('./files')
FIGUREPATH = Path('./results')

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True

def plot_tsne(X_embedded,label,label_str, color_str):
    plt.figure(figsize=(2.6,2.6))
    plt.scatter(X_embedded[label==0,0], X_embedded[label==0,1], s = 2, c = color_str[0], label = label_str[0])
    plt.scatter(X_embedded[label==1,0], X_embedded[label==1,1], s = 2, c = color_str[1 ], label = label_str[1])
    plt.legend(frameon=False)
    plt.tight_layout()

def plotActivity(x, legend_use, color_use='Reds'):
    '''plot activity heatmap'''
    plt.figure(figsize=(2.4,2.4))
    ax = sns.heatmap(x.T, cmap = color_use)
    ax.set_xticks(np.arange(0,x.shape[0]+1, 50))
    ax.set_xticklabels(np.arange(0,x.shape[0]+1, 50), rotation=0)
    ax.set_yticks([0, x.shape[1]-1])
    ax.set_yticklabels([1, x.shape[1]], rotation=0)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Neuron Index')
    ax.set_title(legend_use)
    plt.tight_layout()
    plt.show()
    
def decodingOverlapW():
    config = [10,50,75,100,125,150]
    tsteps = 200
    acc_rule_pfc_all = np.zeros([len(config),round(tsteps/2)])
    acc_rule_md_all = np.zeros([len(config),round(tsteps/2)])
    acc_context_pfc_all = np.zeros([len(config),round(tsteps/2)])
    acc_context_md_all = np.zeros([len(config),round(tsteps/2)])
    
    for i,itau in enumerate(config):
        pickle_in = open('files/final/test_overlapW'+str(itau)+'_numMD'+str(10)+'_numContext'+str(2)+'_MD'+str(True)+'_R'+str(1)+'.pkl','rb')
        data = pickle.load(pickle_in)
        
        cues_all = data['cues_all']
        #cues_all = cues_all[:,:,:4]
        routs_all = data['PFCouts_all']
        MDouts_all = data['MDouts_all']
        
        MDouts_all += np.random.normal(0,0.01,size = MDouts_all.shape)
        
        [num_trial,tsteps,num_cues] = cues_all.shape
        # context
        rule_label = np.zeros(num_trial)
        for i_time in range(num_trial):
            
            temp = sum(cues_all[i_time,0,0::2])
            
            if temp==1:
                rule_label[i_time] = 0
            else:
                rule_label[i_time] = 1
        context_label = np.zeros(num_trial)   
        for i_time in range(num_trial):
            
            temp = np.array( [sum(cues_all[i_time,0,i:i+2]) for i in range(0,len(cues_all[i_time,0,:]),2)])
            temp_index = np.where(temp==1)
            context_label[i_time] = temp_index[0]
        
        ## decode rule from pfc
        acc_rule_pfc = list()
        n_train = int(0.8 * num_trial)
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(routs_all[:,i_time,:],rule_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_rule_pfc.append(score)
        
        ## decode rule from md
        acc_rule_md = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(MDouts_all[:,i_time,:],rule_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            #import pdb;pdb.set_trace()
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_rule_md.append(score)
            
        ## decode context from pfc
        acc_context_pfc = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(routs_all[:,i_time,:],context_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_context_pfc.append(score)
            
        ## decode context from md
        acc_context_md = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(MDouts_all[:,i_time,:],context_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_context_md.append(score)
        
        acc_rule_pfc_all[i,:] = acc_rule_pfc
        acc_rule_md_all[i,:] = acc_rule_md
        acc_context_pfc_all[i,:] = acc_context_pfc
        acc_context_md_all[i,:] = acc_context_md
    
    plot_decoding_vs_para = True
    if plot_decoding_vs_para == True:
        plt.figure(figsize=(2.4,2.4))
        pfc_mean = np.mean(acc_context_pfc_all[:,10:50:10],axis=1)
        md_mean = np.mean(acc_context_md_all[:,10:50:10],axis=1)
        pfc_std = np.std(acc_context_pfc_all[:,10:50:10],axis=1)
        md_std = np.std(acc_context_md_all[:,10:50:10],axis=1)
        plt.plot(config,pfc_mean,'-v',color='tab:red',label='PFC')
        plt.plot(config,md_mean,'-s',color='tab:blue',label='MD')
        plt.fill_between(config, pfc_mean - pfc_std,np.clip(pfc_mean + pfc_std,0,1), alpha=0.2,color='tab:red')
        plt.fill_between(config, md_mean - md_std,np.clip(md_mean + md_std,0,1), alpha=0.2,color='tab:blue')
        plt.legend(frameon=False)
        plt.xlabel('Overlap Size') 
        plt.title('Cue Period')
        plt.ylabel('Decoding Context')
        plt.tight_layout()
        plt.savefig(FIGUREPATH/'pfc_overlapW_decoding_cue.pdf')
        
        plt.figure(figsize=(2.4,2.4))
        pfc_mean = np.mean(acc_context_pfc_all[:,60:100:10],axis=1)
        md_mean = np.mean(acc_context_md_all[:,60:100:10],axis=1)
        pfc_std = np.std(acc_context_pfc_all[:,60:100:10],axis=1)
        md_std = np.std(acc_context_md_all[:,60:100:10],axis=1)
        plt.plot(config,pfc_mean,'-v',color='tab:red',label='PFC')
        plt.plot(config,md_mean,'-s',color='tab:blue',label='MD')
        plt.fill_between(config, pfc_mean - pfc_std,np.clip(pfc_mean + pfc_std,0,1), alpha=0.2,color='tab:red')
        plt.fill_between(config, md_mean - md_std,np.clip(md_mean + md_std,0,1), alpha=0.2,color='tab:blue')
        plt.legend(frameon=False)
        plt.xlabel('Overlap Size')
        plt.ylabel('Decoding Context')
        plt.title('Delay Period')
        plt.tight_layout()
        plt.savefig(FIGUREPATH/'pfc_overlapW_decoding_delay.pdf')
    
    
    plt.figure(figsize=(2.4,2.4))
    plt.plot(acc_rule_pfc,'tab:red',label='PFC')
    plt.plot(acc_rule_md,'tab:blue',label='MD')
    plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
    plt.xticks(np.arange(0,101,20),np.arange(0,201,40))
    plt.xlabel('Time Steps')
    plt.legend(frameon=False)
    plt.ylabel('Decoding Rule')
    #plt.ylim([-0.05, 1.25])
    plt.tight_layout()
    #plt.savefig(FIGUREPATH/'decoding_rule_noiseN.pdf') 
#    plt.savefig(FIGUREPATH/'decoding_rule.png', dpi=300) 
    
#    acc_context_pfc = acc_context_pfc_all[5,:]
#    acc_context_md = acc_context_md_all[5,:]
    plt.figure(figsize=(2.4,2.4))
    plt.plot(acc_context_pfc,'tab:red',label='PFC')
    plt.plot(acc_context_md,'tab:blue',label='MD')
    plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
    plt.xticks(np.arange(0,101,20),np.arange(0,201,40))
    plt.xlabel('Time Steps')
    plt.legend(frameon=False)
    plt.ylabel('Decoding Context')
    #plt.ylim([-0.05, 1.25])
    plt.tight_layout()
    #plt.savefig(FIGUREPATH/'decoding_context_noiseN.pdf') 

def decodingNumMD():
    #config = [10,20,30,40,50]
    config = [2,3,4,5,6]
    tsteps = 200
    acc_rule_pfc_all = np.zeros([len(config),round(tsteps/2)])
    acc_rule_md_all = np.zeros([len(config),round(tsteps/2)])
    acc_context_pfc_all = np.zeros([len(config),round(tsteps/2)])
    acc_context_md_all = np.zeros([len(config),round(tsteps/2)])
    
    for i,itau in enumerate(config):
        pickle_in = open('files/final/test_numMD'+str(12)+'_numContext'+str(itau)+'_MD'+str(True)+'_R'+str(1)+'.pkl','rb')
        data = pickle.load(pickle_in)
        
        cues_all = data['cues_all']
        routs_all = data['PFCouts_all']
        MDouts_all = data['MDouts_all']
        
        MDouts_all += np.random.normal(0,0.01,size = MDouts_all.shape)
        
        [num_trial,tsteps,num_cues] = cues_all.shape
        # context
        rule_label = np.zeros(num_trial)
        for i_time in range(num_trial):
            
            temp = sum(cues_all[i_time,0,0::2])
            
            if temp==1:
                rule_label[i_time] = 0
            else:
                rule_label[i_time] = 1
        context_label = np.zeros(num_trial)   
        for i_time in range(num_trial):
            
            temp = np.array( [sum(cues_all[i_time,0,i:i+2]) for i in range(0,len(cues_all[i_time,0,:]),2)])
            temp_index = np.where(temp==1)
            context_label[i_time] = temp_index[0]
        
        ## decode rule from pfc
        acc_rule_pfc = list()
        n_train = int(0.8 * num_trial)
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(routs_all[:,i_time,:],rule_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_rule_pfc.append(score)
        
        ## decode rule from md
        acc_rule_md = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(MDouts_all[:,i_time,:],rule_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            #import pdb;pdb.set_trace()
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_rule_md.append(score)
            
        ## decode context from pfc
        acc_context_pfc = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(routs_all[:,i_time,:],context_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_context_pfc.append(score)
            
        ## decode context from md
        acc_context_md = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(MDouts_all[:,i_time,:],context_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_context_md.append(score)
        
        acc_rule_pfc_all[i,:] = acc_rule_pfc
        acc_rule_md_all[i,:] = acc_rule_md
        acc_context_pfc_all[i,:] = acc_context_pfc
        acc_context_md_all[i,:] = acc_context_md
    
    plot_decoding_vs_para = True
    if plot_decoding_vs_para == True:
        plt.figure(figsize=(2.4,2.4))
        md_context_mean = np.mean(acc_context_md_all[:,10:50:10],axis=1)
        md_context_std = np.std(acc_context_md_all[:,10:50:10],axis=1)
        md_rule_mean = np.mean(acc_rule_md_all[:,10:50:10],axis=1)
        md_rule_std = np.std(acc_rule_md_all[:,10:50:10],axis=1)  
        plt.plot(config,md_context_mean,'-o',color='tab:orange',label='Context')
        plt.plot(config,md_rule_mean,'-d',color='tab:green',label='Rule')
        plt.fill_between(config, md_context_mean - md_context_std,np.clip(md_context_mean + md_context_std,0,1), alpha=0.2,color='tab:orange')
        plt.fill_between(config, md_rule_mean - md_rule_std,np.clip(md_rule_mean + md_rule_std,0,1), alpha=0.2,color='tab:green')
        plt.legend(frameon=False)
        plt.xlabel('Num Context') 
        plt.title('Cue Period')
        plt.ylabel('Decoding Performance')
        plt.tight_layout()
        plt.savefig(FIGUREPATH/'decoding_numCXT_cue.pdf')
        
        plt.figure(figsize=(2.4,2.4))
        md_context_mean = np.mean(acc_context_md_all[:,60:100:10],axis=1)
        md_context_std = np.std(acc_context_md_all[:,60:100:10],axis=1)
        md_rule_mean = np.mean(acc_rule_md_all[:,60:100:10],axis=1)
        md_rule_std = np.std(acc_rule_md_all[:,60:100:10],axis=1)  
        plt.plot(config,md_context_mean,'-o',color='tab:orange',label='Context')
        plt.plot(config,md_rule_mean,'-d',color='tab:green',label='Rule')
        plt.fill_between(config, md_context_mean - md_context_std,np.clip(md_context_mean + md_context_std,0,1), alpha=0.2,color='tab:orange')
        plt.fill_between(config, md_rule_mean - md_rule_std,np.clip(md_rule_mean + md_rule_std,0,1), alpha=0.2,color='tab:green')
        plt.legend(frameon=False)
        plt.xlabel('Num Context') 
        plt.title('Cue Period')
        plt.ylabel('Decoding Performance')
        plt.tight_layout()
        plt.savefig(FIGUREPATH/'decoding_numCXT_delay.pdf')
        
if __name__ == '__main__':
#    decodingOverlapW()
    decodingNumMD()
    import pdb;pdb.set_trace()
    #Tau_times = [1/2, 1/4, 1/6, 1/8, 1/10]
    # RNGs = [1]
    #config = [1e-2,1e-1,0.3,0.5,0.7,0.9,1e0,1.5,1e1]
    #config = [1e-2,1e-1,0.5,1e0,5,1e1] # input noise std range
    
    #config = [1]
    #Tau_times.extend(range(2,12,2))
    #Hebb_LR = [0,0.0001,0.001,0.01,0.1]
    #num_MD = [10,20,30,40,50]
    tsteps = 200
    acc_rule_pfc_all = np.zeros([len(config),round(tsteps/2)])
    acc_rule_md_all = np.zeros([len(config),round(tsteps/2)])
    acc_context_pfc_all = np.zeros([len(config),round(tsteps/2)])
    acc_context_md_all = np.zeros([len(config),round(tsteps/2)])
    
    plot_pfc_md = False
    
    for i,itau in enumerate(config):
        #pickle_in = open('files/final/test_numMD10_numContext2_MDTrue_R1.pkl','rb')
        pickle_in = open('files/final/test_noiseN'+str(itau)+'_numMD'+str(10)+'_numContext'+str(2)+'_MD'+str(True)+'_R'+str(1)+'.pkl','rb')
        data = pickle.load(pickle_in)
        
        cues_all = data['cues_all']
        #cues_all = cues_all[:,:,:4]
        routs_all = data['PFCouts_all']
        MDouts_all = data['MDouts_all']
        
        MDouts_all += np.random.normal(0,0.01,size = MDouts_all.shape)
        
        [num_trial,tsteps,num_cues] = cues_all.shape
        # context
        rule_label = np.zeros(num_trial)
        for i_time in range(num_trial):
            
            temp = sum(cues_all[i_time,0,0::2])
            
            if temp==1:
                rule_label[i_time] = 0
            else:
                rule_label[i_time] = 1
        context_label = np.zeros(num_trial)   
        for i_time in range(num_trial):
            
            temp = np.array( [sum(cues_all[i_time,0,i:i+2]) for i in range(0,len(cues_all[i_time,0,:]),2)])
            temp_index = np.where(temp==1)
            context_label[i_time] = temp_index[0]
        
        if plot_pfc_md==True:
        
            plotActivity(routs_all[100,:,:],'PFC','Reds')
            #plt.tight_layout()
            plt.savefig(FIGUREPATH/'pfc_ctx1.pdf') 
            #plt.savefig(FIGUREPATH/'pfc_ctx1.png', dpi=300)
            plotActivity(routs_all[200,:,:],'PFC','Reds')
            #plt.tight_layout()
            plt.savefig(FIGUREPATH/'pfc_ctx2.pdf') 
            #plt.savefig(FIGUREPATH/'pfc_ctx2.png', dpi=300)
            
            plotActivity(data['MDouts_all'][100,:,:],'MD','Blues_r')
            #plt.tight_layout()
            plt.savefig(FIGUREPATH/'md_ctx1.pdf') 
            #plt.savefig(FIGUREPATH/'md_ctx1.png: dpi=300)
            plotActivity(data['MDouts_all'][200,:,:],'MD','Blues_r')
            #plt.tight_layout()
            plt.savefig(FIGUREPATH/'md_ctx2.pdf') 
            #plt.savefig(FIGUREPATH/'md_ctx2.png', dpi=300)
        #import pdb;pdb.set_trace()
        plot_pfc_md_tsne = False
        if plot_pfc_md_tsne == True:
            legend_str = ['Rule 1', 'Rule 2']
            color_str = ['tab:red','tab:blue']
            pfc_embedded = TSNE(n_components=2).fit_transform(routs_all[:,120,:])
            plot_tsne(pfc_embedded,rule_label,legend_str,color_str)
            md_embedded = TSNE(n_components=2).fit_transform(MDouts_all[:,120,:])
            plot_tsne(md_embedded,rule_label,legend_str,color_str)
            legend_str = ['Context 1', 'Context 2']
            color_str = ['tab:orange', 'tab:green']
            plot_tsne(pfc_embedded,context_label,legend_str,color_str)
            plot_tsne(md_embedded,context_label,legend_str,color_str)
            
            
        #
        ## decode rule from pfc
        acc_rule_pfc = list()
        n_train = int(0.8 * num_trial)
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(routs_all[:,i_time,:],rule_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            #import pdb;pdb.set_trace()
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_rule_pfc.append(score)
        
        ## decode rule from md
        acc_rule_md = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(MDouts_all[:,i_time,:],rule_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            #import pdb;pdb.set_trace()
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_rule_md.append(score)
            
        ## decode context from pfc
        acc_context_pfc = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(routs_all[:,i_time,:],context_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_context_pfc.append(score)
            
        ## decode context from md
        acc_context_md = list()
        for i_time in range(0,tsteps,2):
            X, y = sklearn.utils.shuffle(MDouts_all[:,i_time,:],context_label)
            X_train, X_test = X[:n_train],X[n_train:]
            y_train, y_test = y[:n_train],y[n_train:]
            
            #clf = LinearSVC()
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            acc_context_md.append(score)
        
        acc_rule_pfc_all[i,:] = acc_rule_pfc
        acc_rule_md_all[i,:] = acc_rule_md
        acc_context_pfc_all[i,:] = acc_context_pfc
        acc_context_md_all[i,:] = acc_context_md
    
#    pickle_out = open('decoding_results_MDlearn_11_Blocks.pickle','wb')
#    pickle.dump({'acc_rule_pfc_all':acc_rule_pfc_all,\
#                 'acc_rule_md_all':acc_rule_md_all,\
#                 'acc_context_pfc_all':acc_context_pfc_all,\
#                 'acc_context_md_all':acc_context_md_all},pickle_out)
#    pickle_out.close()
        ## plot pfc md activity
    
    
    plot_decoding_vs_para = True
    if plot_decoding_vs_para == True:
        plt.figure(figsize=(2.4,2.4))
        pfc_mean = np.mean(acc_context_pfc_all[:,10:50:10],axis=1)
        md_mean = np.mean(acc_context_md_all[:,10:50:10],axis=1)
        pfc_std = np.std(acc_context_pfc_all[:,10:50:10],axis=1)
        md_std = np.std(acc_context_md_all[:,10:50:10],axis=1)
        plt.semilogx(config,pfc_mean,'-v',color='tab:red',label='PFC')
        plt.semilogx(config,md_mean,'-s',color='tab:blue',label='MD')
        plt.fill_between(config, pfc_mean - pfc_std,np.clip(pfc_mean + pfc_std,0,1), alpha=0.2,color='tab:red')
        plt.fill_between(config, md_mean - md_std,np.clip(md_mean + md_std,0,1), alpha=0.2,color='tab:blue')
        plt.legend(frameon=False)
        plt.xlabel('Input Noise STD') 
        plt.title('Cue Period')
        plt.ylabel('Decoding Context')
        plt.tight_layout()
        plt.savefig(FIGUREPATH/'pfc_noiseN_decoding_cue.pdf')
        
        plt.figure(figsize=(2.4,2.4))
        pfc_mean = np.mean(acc_context_pfc_all[:,60:100:10],axis=1)
        md_mean = np.mean(acc_context_md_all[:,60:100:10],axis=1)
        pfc_std = np.std(acc_context_pfc_all[:,60:100:10],axis=1)
        md_std = np.std(acc_context_md_all[:,60:100:10],axis=1)
        plt.semilogx(config,pfc_mean,'-v',color='tab:red',label='PFC')
        plt.semilogx(config,md_mean,'-s',color='tab:blue',label='MD')
        plt.fill_between(config, pfc_mean - pfc_std,np.clip(pfc_mean + pfc_std,0,1), alpha=0.2,color='tab:red')
        plt.fill_between(config, md_mean - md_std,np.clip(md_mean + md_std,0,1), alpha=0.2,color='tab:blue')
        plt.legend(frameon=False)
        plt.xlabel('Input Noise STD')
        plt.ylabel('Decoding Context')
        plt.title('Delay Period')
        plt.tight_layout()
        plt.savefig(FIGUREPATH/'pfc_noiseN_decoding_delay.pdf')
    
    
    plt.figure(figsize=(2.4,2.4))
    plt.plot(acc_rule_pfc,'tab:red',label='PFC')
    plt.plot(acc_rule_md,'tab:blue',label='MD')
    plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
    plt.xticks(np.arange(0,101,20),np.arange(0,201,40))
    plt.xlabel('Time Steps')
    plt.legend(frameon=False)
    plt.ylabel('Decoding Rule')
    #plt.ylim([-0.05, 1.25])
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'decoding_rule.pdf') 
#    plt.savefig(FIGUREPATH/'decoding_rule.png', dpi=300) 
    
#    acc_context_pfc = acc_context_pfc_all[5,:]
#    acc_context_md = acc_context_md_all[5,:]
    plt.figure(figsize=(2.4,2.4))
    plt.plot(acc_context_pfc,'tab:red',label='PFC')
    plt.plot(acc_context_md,'tab:blue',label='MD')
    plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
    plt.xticks(np.arange(0,101,20),np.arange(0,201,40))
    plt.xlabel('Time Steps')
    plt.legend(frameon=False)
    plt.ylabel('Decoding Context')
    #plt.ylim([-0.05, 1.25])
    plt.tight_layout()
    plt.savefig(FIGUREPATH/'decoding_context.pdf') 
#    plt.savefig(FIGUREPATH/'decoding_context.png', dpi=300) 

#plt.figure(figsize=(2.4,2.4))
#plt.plot(acc_rule_md,'tab:red',label='Rule')
#plt.plot(acc_context_md,'tab:blue',label='Context')
#plt.fill_between(range(0,50),1*np.ones(50),0.28*np.ones(50),facecolor='orange',alpha=0.2)
#plt.xticks(np.arange(0,101,20),np.arange(0,201,40))
#plt.xlabel('Time Steps')
#plt.legend(frameon=False)
#plt.ylabel('Accuracy')
##plt.ylim([-0.05, 1.25])
#plt.tight_layout()
#plt.savefig(FIGUREPATH/'decoding_md_noiseN.pdf')
    
#    import pdb;pdb.set_trace()
#    run_decoder(path)
#    plot_decoder(path)
    # activity, info = load_activity(path)
#
#plt.plot(acc_context_md_all[0,:],'-r',label='25 MD')
#plt.legend()
#plt.figure()
#plt.plot(acc_context_md_all[0,:],'-r',label='8 MD')
#plt.legend()
##plt.plot(acc_context_md_all[2,:],'-g',label='75 MD')
##plt.legend()
#plt.plot(acc_context_md_all[1,:],'-b',label='16 MD')
#plt.legend()
#plt.xlabel('Time steps')
#plt.ylabel('Decoding Context (MD)')
#plt.fill_between(range(0,100),1*np.ones(100),0.4*np.ones(100),facecolor='orange',alpha=0.2)
#
#    
#plt.plot(acc_rule_pfc_all.T)
#plt.plot(acc_rule_md_all.T)
#plt.xlabel('Time steps')
#plt.ylabel('Decoding Rule')
#plt.fill_between(range(0,50),1*np.ones(50),0.4*np.ones(50),facecolor='orange',alpha=0.2)
#
#plt.plot(acc_context_pfc_all.T)
#plt.plot(acc_context_md_all.T)
#plt.xlabel('Time steps')
#plt.ylabel('Decoding Context')
#plt.fill_between(range(0,50),1*np.ones(50),0.4*np.ones(50),facecolor='orange',alpha=0.2)
    