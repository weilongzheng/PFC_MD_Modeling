'''
Use heatmap to visualize connection weights
'''

import numpy as np
from collections import defaultdict
from pathlib import Path
import os
import sys
import pickle
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from temp_task import RikhyeTask
from temp_model import PytorchPFCMD
import matplotlib.pyplot as plt
import seaborn as sns



# Generate trainset
RNGSEED = 5 # set random seed
np.random.seed([RNGSEED])

Ntrain = 200            # number of training cycles for each context
Nextra = 200            # add cycles to show if block1
Ncontexts = 2           # number of cueing contexts (e.g. auditory cueing context)
inpsPerConext = 2       # in a cueing context, there are <inpsPerConext> kinds of stimuli
                         # (e.g. auditory cueing context contains high-pass noise and low-pass noise)

# Model settings
n_neuron = 1000
n_neuron_per_cue = 200
Num_MD = 10
num_active = 5  # num MD active per context
n_output = 2
MDeffect = True
PFClearn = False
shift = 1 # shift step list


# Get connection weights
filename = Path('files')
os.makedirs(filename, exist_ok=True)
file_training = 'train_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_shift'+str(shift)+'_R'+str(RNGSEED)+'.pkl'
with open(filename / file_training, 'rb') as f:
    log = pickle.load(f)

wPFC2MD = log['wPFC2MD']
wMD2PFC = log['wMD2PFC']


# Heatmap wPFC2MD
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(wPFC2MD, cmap='bwr')
ax.set_xticks([0, 999])
ax.set_xticklabels([1, 1000], rotation=0)
ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_xlabel('PFC neuron index')
ax.set_ylabel('MD neuron index')
ax.set_title('wPFC2MD '+'PFC learnable-'+str(PFClearn)+' shift step-'+str(shift))
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight')

# Heatmap wMD2PFC
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(wMD2PFC, cmap='bwr')
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_yticks([0, 999])
ax.set_yticklabels([1, 1000], rotation=0)
ax.set_xlabel('MD neuron index')
ax.set_ylabel('PFC neuron index')
ax.set_title('wMD2PFC '+'PFC learnable-'+str(PFClearn)+' shift step-'+str(shift))
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight')