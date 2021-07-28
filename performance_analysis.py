# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 18:38:46 2021

@author: weilong
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from pathlib import Path

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True

filename = Path('files\\final')
Num_MD = 10
Ncontexts = 2
PFClearn = False
seed_setup = [1,2,3,4,5,6,7,8,9,10]
#seed_setup = [1,3,5]

MDeffect = True
mse_md = list()
for RNGSEED in seed_setup:
    file = 'train_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_R'+str(RNGSEED)+'.pkl'
    file = open(filename / file,'rb')
    data = pickle.load(file)
    mse_md.append(data['mse'])
    
mse_md_mean = np.mean(mse_md,axis=0)
#import pdb;pdb.set_trace()
MDeffect = False
mse_mdoff = list()
for RNGSEED in seed_setup:
    file = 'train_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_R'+str(RNGSEED)+'.pkl'
    file = open(filename / file,'rb')
    data = pickle.load(file)
    mse_mdoff.append(data['mse'])
    
mse_mdoff_mean = np.mean(mse_mdoff,axis=0)

filesave = Path('results')
os.makedirs(filesave, exist_ok=True)

plt.figure(figsize=(2.4,2.4))
plt.plot(mse_mdoff_mean,'tab:red',label='Without MD')
#ci_off = 1.96 * np.std(mse_mdoff,axis=0)/np.mean(mse_mdoff,axis=0)
ci_off = np.std(mse_mdoff,axis=0)
plt.fill_between(np.arange(len(mse_mdoff_mean)), (mse_mdoff_mean-ci_off), (mse_mdoff_mean+ci_off), color='tab:red', alpha=.2)
plt.plot(mse_md_mean,'tab:blue',label='With MD')
#ci_on = 1.96 * np.std(mse_md,axis=0)/np.mean(mse_md,axis=0)
ci_on = np.std(mse_md,axis=0)
plt.fill_between(np.arange(len(mse_md_mean)), (mse_md_mean-ci_on), (mse_md_mean+ci_on), color='tab:blue', alpha=.2)
plt.xticks(np.arange(0,301,100),np.arange(0,601,200))
plt.xlabel('Trials'),plt.ylabel('MSE')
plt.ylim(0, 0.6)
plt.legend(frameon=False)
plt.axvspan(0, 100, ymin=0, ymax=1, alpha=0.1, color='tab:orange')
plt.axvspan(200, 300, ymin=0, ymax=1, alpha=0.1, color='tab:orange')
plt.axvspan(100, 200, ymin=0, ymax=1, alpha=0.1, color='tab:green')
plt.title('Model Performance')
plt.tight_layout()
#plt.savefig(filesave/'mse.pdf') 
plt.savefig(filesave/'mse.pdf', dpi=300) 

plt.figure(figsize=(2.4,2.4))
plt.plot(mse_mdoff_mean[200:],'tab:red',label='Without MD')
ci_off = np.std(mse_mdoff,axis=0)
plt.fill_between(np.arange(len(mse_mdoff_mean[200:])), (mse_mdoff_mean[200:]-ci_off[200:]), (mse_mdoff_mean[200:]+ci_off[200:]), color='tab:red', alpha=.2)
plt.plot(mse_md_mean[200:],'tab:blue',label='With MD')
ci_on = np.std(mse_md,axis=0)
plt.fill_between(np.arange(len(mse_md_mean[200:])), (mse_md_mean[200:]-ci_on[200:]), (mse_md_mean[200:]+ci_on[200:]), color='tab:blue', alpha=.2)
plt.xticks(np.arange(0,101,25),np.arange(400,601,50))
plt.xlabel('Trials'),plt.ylabel('MSE')
plt.legend(frameon=False)
plt.title('Model Performance')
plt.tight_layout()
plt.savefig(filesave/'mse_switch.pdf', dpi=300) 
#"""
#MSE vs. cycles
#"""

#import pickle
#import os
#import matplotlib.pyplot as plt
#import numpy as np
#import matplotlib as mpl
#from pathlib import Path
#
#mpl.rcParams['font.size'] = 7
#mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['ps.fonttype'] = 42
#mpl.rcParams['font.family'] = 'arial'
#
#filename = Path('files')
#Num_MD = 10
#Ncontexts = 2
#PFClearn = False
#seed_setup = [1,3,5]
#
#MDeffect = True
#mse_md = list()
#for RNGSEED in seed_setup:
#    file = 'train_softSwitch_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_R'+str(RNGSEED)+'.pkl'
#    file = open(filename / file,'rb')
#    data = pickle.load(file)
#    mse_md.append(data['mse'])
#    
#mse_md = np.mean(mse_md,axis=0)*2
##import pdb;pdb.set_trace()
#MDeffect = False
#mse_mdoff = list()
#for RNGSEED in seed_setup:
#    file = 'train_softSwitch_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_R'+str(RNGSEED)+'.pkl'
#    file = open(filename / file,'rb')
#    data = pickle.load(file)
#    mse_mdoff.append(data['mse'])
#    
#mse_mdoff = np.mean(mse_mdoff,axis=0)*2
#
#filesave = Path('results')
#os.makedirs(filesave, exist_ok=True)
#
#plt.figure(figsize=(3,2.4))
#plt.plot(mse_mdoff,'tab:red',label='Without MD')
#plt.plot(mse_md,'tab:blue',label='With MD')
#plt.xticks(np.arange(0,301,100),np.arange(0,301,100))
#plt.xlabel('Cycles'),plt.ylabel('MSE')
#plt.legend(frameon=False)
#plt.tight_layout()
#plt.axvspan(0, 100, ymin=0, ymax=1, alpha=0.2, color='tab:orange')
#plt.axvspan(200, 300, ymin=0, ymax=1, alpha=0.2, color='tab:orange')
#plt.axvspan(100, 200, ymin=0, ymax=1, alpha=0.2, color='tab:green')
##plt.savefig(filesave/'mse.pdf') 
#plt.savefig(filesave/'mse.png', dpi=300) 