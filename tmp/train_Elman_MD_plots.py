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



model_name = 'Elman_MD'+'_MDeffect'+str(MDeffect)+'_Sensoryinputlearn'+str(Sensoryinputlearn)+\
                '_Elmanlearn'+str(Elmanlearn)+'_R'+str(RNGSEED)

with open(directory / (model_name + '.pkl'), 'rb') as f:
    log = pickle.load(f)

# Plot MSE curve
plt.plot(log['mse'], label='Elman MD')
plt.xlabel('Cycles')
plt.ylabel('MSE loss')
plt.legend()
#plt.xticks([0, 500, 1000, 1200])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#plt.tight_layout()
plt.show()



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