import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import gym
import neurogym as ngym
from model_dev import PytorchPFCMD

import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from pygifsicle import optimize


# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plt.plot(log['losses'])
plt.xlabel('Training Cycles', fontdict=font)
plt.ylabel('CE loss', fontdict=font)
# plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
# plt.ylim([0.0, 1.0])
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
plt.savefig('./animation/'+'CEloss.png')
plt.show()

# Heatmap wPFC2MD
font = {'family':'Times New Roman','weight':'normal', 'size':30}
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(model.md.wPFC2MD, cmap='Reds')
ax.set_xticks([0, 999])
ax.set_xticklabels([1, 1000], rotation=0)
ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_xlabel('PFC neuron index', fontdict=font)
ax.set_ylabel('MD neuron index', fontdict=font)
ax.set_title('wPFC2MD', fontdict=font)
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight', fontdict=font)
plt.tight_layout()
plt.savefig('./animation/'+'wPFC2MD.png')
plt.show()

# Heatmap wMD2PFC
font = {'family':'Times New Roman','weight':'normal', 'size':30}
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(model.md.wMD2PFC, cmap='Blues_r')
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_yticks([0, 999])
ax.set_yticklabels([1, 1000], rotation=0)
ax.set_xlabel('MD neuron index', fontdict=font)
ax.set_ylabel('PFC neuron index', fontdict=font)
ax.set_title('wMD2PFC', fontdict=font)
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight', fontdict=font)
plt.tight_layout()
plt.savefig('./animation/'+'wMD2PFC.png')
plt.show()

# MD pretraces evolution
font = {'family':'Times New Roman','weight':'normal', 'size':20}
plot_step = 10

meanMDpreTraces_all = np.mean(log['MDpreTraces_all'], axis=1)
meanMDpreTrace_threshold_all = np.mean(log['MDpreTrace_threshold_all'], axis=1)
for i in range(len(log['MDpreTraces_all'])):
    if (i+1) % plot_step == 0:
        MDpreTraces = meanMDpreTraces_all[i, :]
        MDpreTrace_threshold = meanMDpreTrace_threshold_all[i, :]
        plt.plot(MDpreTraces)  
        plt.axhline(y=MDpreTrace_threshold, color='r', linestyle='-')
        plt.title('MD pretraces' + ' Cycle-'+str(i+1), fontdict=font)
        plt.xlabel('PFC neuron index')
        plt.ylabel('Presynaptic activities')
        plt.xlim([-50, 1050])
        plt.ylim([0.0, 1.0])
        plt.xticks(ticks=[100*i for i in range(11)], rotation=0)
        plt.yticks(ticks=[0.1*i for i in range(11)], rotation=0)
        plt.savefig('./animation/'+f'MDpreTraces_index_{i}.png')
        plt.close() # do not show figs in line

images = []
for i in range(len(log['MDpreTraces_all'])):
    if (i+1) % plot_step == 0:
        filename = './animation/'+f'MDpreTraces_index_{i}.png'
        images.append(imageio.imread(filename))
gif_path = './animation/'+'MDpreTraces_evolution.gif'
imageio.mimsave(gif_path, images, duration=0.5)
optimize(gif_path)

# MD outputs evolution
font = {'family':'Times New Roman','weight':'normal', 'size':20}
plot_step = 10

meanMDouts_all = np.mean(log['MDouts_all'], axis=1)
for i in range(len(log['MDouts_all'])):
    if (i+1) % plot_step == 0:
        meanMDouts = meanMDouts_all[i, :]
        plt.plot(meanMDouts)
        plt.title('MD outputs' + ' Cycle-'+str(i+1), fontdict=font)
        plt.xlabel('MD neuron index')
        plt.ylabel('MD activities')
        plt.xlim([0, 9])
        plt.ylim([-0.05, 1.05])
        plt.xticks(ticks=[i for i in range(10)], labels=[i+1 for i in range(10)], rotation=0)
        plt.yticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], rotation=0)
        plt.savefig('./animation/'+f'MDoutputs_index_{i}.png')
        plt.close() # do not show figs in line

images = []
for i in range(len(log['MDouts_all'])):
    if (i+1) % plot_step == 0:
        filename = './animation/'+f'MDoutputs_index_{i}.png'
        images.append(imageio.imread(filename))
gif_path = './animation/'+'MDoutputs_evolution.gif'
imageio.mimsave(gif_path, images, duration=0.5)
optimize(gif_path)