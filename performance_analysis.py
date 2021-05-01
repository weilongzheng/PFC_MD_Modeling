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

filename = Path('files')
Num_MD = 10
Ncontexts = 2
PFClearn = True
seed_setup = [1,10,5]

MDeffect = True
mse_md = list()
for RNGSEED in seed_setup:
    file = 'train_multicues_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_R'+str(RNGSEED)+'.pkl'
    file = open(filename / file,'rb')
    data = pickle.load(file)
    mse_md.append(data['mse'])
    
mse_md = np.mean(mse_md,axis=0)
# import pdb;pdb.set_trace()
MDeffect = False
mse_mdoff = list()
for RNGSEED in seed_setup:
    file = 'train_multicues_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_R'+str(RNGSEED)+'.pkl'
    file = open(filename / file,'rb')
    data = pickle.load(file)
    mse_mdoff.append(data['mse'])
    
mse_mdoff = np.mean(mse_mdoff,axis=0)

filesave = Path('results')
os.makedirs(filesave, exist_ok=True)

plt.figure(figsize=(3,2.4))
plt.plot(mse_mdoff,'tab:red',label='Without MD')
plt.plot(mse_md,'tab:blue',label='With MD')
plt.xticks(np.arange(0,601,200),np.arange(0,1201,400))
plt.xlabel('Trials'),plt.ylabel('MSE')
plt.legend(frameon=False)
plt.tight_layout()
plt.axvspan(0, 200, ymin=0, ymax=1, alpha=0.2, color='tab:orange')
plt.axvspan(400, 600, ymin=0, ymax=1, alpha=0.2, color='tab:orange')
plt.axvspan(200, 400, ymin=0, ymax=1, alpha=0.2, color='tab:green')
#plt.savefig(filesave/'mse.pdf') 
plt.savefig(filesave/'mse_allnoise.png', dpi=300) 