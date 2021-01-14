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


#def map_to_class(x):
#    """Map array to integer classes."""
#    unique_val = np.unique(x)
#    y = np.zeros_like(x)
#    for i, v in enumerate(unique_val):        
#        y[x==v] = i
#    return y
#
#
#def decode(activity, info, area, decode_var):
#    """Decode classes in time."""
#    n_train = int(0.8 * activity[area].shape[0])
#    X_train, X_test = activity[area][:n_train], activity[area][n_train:]
#    
#    # Adding noise
#    noise_scale = 0.1
#    X_train += np.random.randn(*X_train.shape) * noise_scale
#    X_test += np.random.randn(*X_test.shape) * noise_scale
#
#    y = map_to_class(info[decode_var])
#    y_train, y_test = y[:n_train], y[n_train:]
#    
#    from sklearn.svm import LinearSVC
#    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#    # clf = LinearSVC()
#    clf = LinearDiscriminantAnalysis()
#    
#    scores = list()
#    for i_time in range(X_train.shape[1]):
#        clf.fit(X_train[:, i_time, :], y_train)
#        # score = clf.score(X_train[:, i_time, :], y_train)
#        score = clf.score(X_test[:, i_time, :], y_test)
#        scores.append(score)
#    return scores
#
#
#def run_decoder(path):
#    """Run decoder across areas and variables."""
#    activity, info = load_activity(path)
#    areas = list(activity.keys())
#    decode_vars = info.keys()
#    decoding_results = pd.DataFrame()
#    for area in areas:
#        for decode_var in decode_vars:
#            if len(pd.unique(info[decode_var])) > 1:
#                scores = decode(activity, info, area, decode_var)
#                res = {'area': area, 'decode_var': decode_var, 'scores': scores}
#                decoding_results = decoding_results.append(res, ignore_index=True)
#            else:
#                print('Only exist 1 value for ', decode_var, ' at ',
#                      pd.unique(info[decode_var]))
#    
#    decoding_results.to_pickle(MODELPATH / path / 'decoding.pkl')
#
#
#def plot_decoder(path):
#    """Plot decoder results."""
#    cfg = configs.BaseConfig()
#    cfg.load(MODELPATH / path / 'cfg.json')
#    dt = cfg.task.kwargs.dt
#    
#    res = pd.read_pickle(MODELPATH / path / 'decoding.pkl')
#    decode_vars = pd.unique(res['decode_var'])
#    areas = pd.unique(res['area'])
#    
#    def _plot(decode_var):
#        plt.figure(figsize=(3, 1.5))
#        t_plot = np.arange(len(res.loc[0]['scores']))*dt
#        for area in areas:
#            selection = np.where((res['area']==area) & (res['decode_var']==decode_var))[0][0]
#            plt.plot(t_plot, res.loc[selection]['scores'], label=area)
#        plt.xlabel('Time (ms)')
#        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#        plt.ylabel('Decoding ' + decode_var)
#        plt.ylim([-0.05, 1.25])
#        plt.tight_layout()
#        
#        annotate_schmitt(plt, ytext=1.1)
#        
#        plt.savefig(FIGUREPATH / path / ('decode_' + decode_var + '.png'),
#                    transparent=True, dpi=600)
#        # plt.savefig(FIGUREPATH / path / ('decode_' + decode_var + '.pdf'))
#
#    for decode_var in decode_vars:
#        _plot(decode_var)
#
#
#def run_plot_distance(path):
#    activity, info = load_activity(path)
#    areas = list(activity.keys())
#    decode_vars = info.keys()
#    
#    dist_results = pd.DataFrame()
#    for area in areas:
#        for decode_var in decode_vars:
#            if len(pd.unique(info[decode_var])) == 2:
#                unique_values = pd.unique(info[decode_var])
#                dist = (activity[area][info[decode_var]==unique_values[0]].mean(axis=0) -
#                        activity[area][info[decode_var]==unique_values[1]].mean(axis=0))**2
#                dist = dist.mean(axis=-1)
#                res = {'area': area, 'decode_var': decode_var, 'dist': dist}
#                dist_results = dist_results.append(res, ignore_index=True)
#            else:
#                print('Have not-2 values for ', decode_var, ' at ',
#                      pd.unique(info[decode_var]))
#
#    res = dist_results
#    cfg = configs.BaseConfig()
#    cfg.load(MODELPATH / path / 'cfg.json')
#    dt = cfg.task.kwargs.dt
#
#    def _plot(decode_var):
#        plt.figure(figsize=(3, 1.5))
#        t_plot = np.arange(len(res.loc[0]['dist']))*dt
#        maxval = -np.inf
#        for area in areas:
#            selection = np.where((res['area']==area) & (res['decode_var']==decode_var))[0][0]
#            plt.plot(t_plot, res.loc[selection]['dist'], label=area)
#            maxval = max(maxval, np.max(res.loc[selection]['dist']))
#        plt.xlabel('Time (ms)')
#        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#        plt.ylabel('Distance ' + decode_var)
#        plt.ylim([-0.05*maxval, 1.25*maxval])
#        plt.tight_layout()
#        
#        annotate_schmitt(plt, ytext=1.1 * maxval)
#        
#        # plt.savefig(FIGUREPATH / path / ('decode_' + decode_var + '.png'),
#        #             transparent=True, dpi=600)
#        # plt.savefig(FIGUREPATH / path / ('decode_' + decode_var + '.pdf'))
#
#    for decode_var in pd.unique(res['decode_var']):
#        _plot(decode_var)


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
        pickle_in = open('dataPFCMD/test_HebbPostTrace_numMD9_numTask3_MDTrue_LearnTrue_R5_TimesTau4.pickle','rb')
#        pickle_in = open('dataPFCMD/test'+\
#                         '_MDTrue_LearnTrue_R5_'+\
#                         'TimesTau'+str(itau)+'.pickle','rb')
        data = pickle.load(pickle_in)
        
        cues_all = data['cues_all']
        routs_all = data['routs_all']
        MDouts_all = data['MDouts_all']
        MDinps_all = data['MDinps_all']
        outs_all = data['outs_all']
        
        MDouts_all += np.random.normal(0,0.01,size = MDouts_all.shape)
        
        [num_trial,tsteps,num_cues] = cues_all.shape
        # 3 context
        rule_label = np.zeros(num_trial)
        for i_time in range(num_trial):
            if cues_all[i_time,0,0]==1 or cues_all[i_time,0,2]==1 or cues_all[i_time,0,4]==1:
                rule_label[i_time] = 0
            else:
                rule_label[i_time] = 1
        context_label = np.zeros(num_trial)   
        for i_time in range(num_trial):
            if cues_all[i_time,0,0]==1 or cues_all[i_time,0,1]==1:
                context_label[i_time] = 0
            elif cues_all[i_time,0,2]==1 or cues_all[i_time,0,3]==1:
                context_label[i_time] = 1
            else:
                context_label[i_time] = 2
        # 2 context
#        rule_label = np.zeros(num_trial)
#        for i_time in range(num_trial):
#            if cues_all[i_time,0,0]==1 or cues_all[i_time,0,2]==1:
#                rule_label[i_time] = 0
#            else:
#                rule_label[i_time] = 1
#        context_label = np.zeros(num_trial)   
#        for i_time in range(num_trial):
#            if cues_all[i_time,0,0]==1 or cues_all[i_time,0,1]==1:
#                context_label[i_time] = 0
#            else:
#                context_label[i_time] = 1
        
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
    
#plt.figure()
#plt.plot(acc_rule_pfc,'-r',label='PFC')
#plt.legend()
#plt.plot(acc_rule_md,'-b',label='MD')
#plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
#plt.xlabel('Time steps')
#plt.legend()
#plt.ylabel('Decoding Rule')
##plt.ylim([-0.05, 1.25])
#plt.tight_layout()
#    
#plt.figure()
#plt.plot(acc_context_pfc,'-r',label='PFC')
#plt.legend()
#plt.plot(acc_context_md,'-b',label='MD')
#plt.fill_between(range(0,50),1*np.ones(50),0.3*np.ones(50),facecolor='orange',alpha=0.2)
#plt.xlabel('Time steps')
#plt.legend()
#plt.ylabel('Decoding Context')
##plt.ylim([-0.05, 1.25])
#plt.tight_layout()
    
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
    