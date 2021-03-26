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

MODELPATH = Path('./files')
FIGUREPATH = Path('./figures')


if __name__ == '__main__':
    #Tau_times = [1/2, 1/4, 1/6, 1/8, 1/10]
    Tau_times = [1]
    #Tau_times.extend(range(2,12,2))
    #Hebb_LR = [0,0.0001,0.001,0.01,0.1]
    #num_MD = [10,20,30,40,50]
    tsteps = 200
    acc_rule_pfc_all = np.zeros([len(Tau_times),round(tsteps/2)])
    acc_rule_md_all = np.zeros([len(Tau_times),round(tsteps/2)])
    acc_context_pfc_all = np.zeros([len(Tau_times),round(tsteps/2)])
    acc_context_md_all = np.zeros([len(Tau_times),round(tsteps/2)])
    
    for i,itau in enumerate(Tau_times):
        pickle_in = open('files/train_numMD10_numContext2_MDTrue_R5.pkl','rb')
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
    
plt.figure()
plt.plot(acc_rule_pfc,'-r',label='PFC')
plt.legend()
plt.plot(acc_rule_md,'-b',label='MD')
plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
plt.xticks(np.arange(0,101,10),np.arange(0,201,20))
plt.xlabel('Time steps')
plt.legend()
plt.ylabel('Decoding Rule')
#plt.ylim([-0.05, 1.25])
plt.tight_layout()
    
plt.figure()
plt.plot(acc_context_pfc,'-r',label='PFC')
plt.legend()
plt.plot(acc_context_md,'-b',label='MD')
plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
plt.xticks(np.arange(0,101,10),np.arange(0,201,20))
plt.xlabel('Time steps')
plt.legend()
plt.ylabel('Decoding Context')
#plt.ylim([-0.05, 1.25])
plt.tight_layout()
    
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
    