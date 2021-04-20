import numpy as np
from collections import defaultdict
from pathlib import Path
import os
import sys
import pickle
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from task import RikhyeTask
from model import PytorchPFCMD
import matplotlib.pyplot as plt
import seaborn as sns


model_name = 'Elman_MD'+'_MDeffect'+str(MDeffect)+'_Sensoryinputlearn'+str(Sensoryinputlearn)+\
                '_Elmanlearn'+str(Elmanlearn)+'_R'+str(RNGSEED)


with open(directory / (model_name + '.pkl'), 'rb') as f:
    log = pickle.load(f)

font = {'family':'Times New Roman','weight':'normal', 'size':24}

# Plot total loss curve
plt.plot(log['loss_val'], label='Elman MD')
plt.xlabel('Cycles', fontdict=font)
plt.ylabel('Loss value', fontdict=font)
plt.legend()
#plt.xticks([0, 500, 1000, 1200])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#plt.tight_layout()
plt.show()

# Plot MSE curve
plt.plot(log['mse'], label='Elman MD')
plt.xlabel('Cycles', fontdict=font)
plt.ylabel('MSE Loss', fontdict=font)
plt.legend()
#plt.xticks([0, 500, 1000, 1200])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#plt.tight_layout()
plt.show()


# Plot Elman connection weights
Winput2h = log['Winput2h']
Wrec = log['Wrec']

## Heatmap Winput2h
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(Winput2h, cmap='bwr')
ax.set_xticklabels([1, 2, 3, 4], rotation=0)
ax.set_yticks([0, 999])
ax.set_yticklabels([1, 1000], rotation=0)
ax.set_xlabel('Cue index', fontdict=font)
ax.set_ylabel('Elman neuron index', fontdict=font)
ax.set_title('Weights: input to hiddenlayer', fontdict=font)
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight', fontdict=font)
plt.show()

## Heatmap Wrec
plt.figure(figsize=(15, 15))
plt.matshow(Wrec, cmap='bwr')
plt.xlabel('Elman neuron index', fontdict=font)
plt.ylabel('Elman neuron index', fontdict=font)
plt.xticks(ticks=[0, 999], labels=[1, 1000])
plt.yticks(ticks=[0, 999], labels=[1, 1000])
plt.title('Weights: recurrent', fontdict=font)
plt.show()



# Plot MD connection weights

wPFC2MD = log['wPFC2MD']
wMD2PFC = log['wMD2PFC']

## Heatmap wPFC2MD
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(wPFC2MD, cmap='bwr')
ax.set_xticks([0, 999])
ax.set_xticklabels([1, 1000], rotation=0)
ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_xlabel('Elman neuron index', fontdict=font)
ax.set_ylabel('MD neuron index', fontdict=font)
ax.set_title('wElman2MD', fontdict=font)
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight', fontdict=font)

## Heatmap wMD2PFC
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(wMD2PFC, cmap='bwr')
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_yticks([0, 999])
ax.set_yticklabels([1, 1000], rotation=0)
ax.set_xlabel('Elman neuron index', fontdict=font)
ax.set_ylabel('PFC neuron index', fontdict=font)
ax.set_title('wMD2Elman', fontdict=font)
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight', fontdict=font)