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


# wPFC2MD evolution
font = {'family':'Times New Roman','weight':'normal', 'size':30}
wPFC2MD_max = 0
for i in range(len(log['wPFC2MD_list'])):
    wPFC2MD = log['wPFC2MD_list'][i]
    if  wPFC2MD_max < wPFC2MD.max():
        wPFC2MD_max = wPFC2MD.max()

for i in range(len(log['wPFC2MD_list'])):
    wPFC2MD = log['wPFC2MD_list'][i]
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(wPFC2MD, cmap='Reds', vmax=wPFC2MD_max, vmin=0.0)
    ax.set_xticks([0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999])
    ax.set_xticklabels([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], rotation=0)
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_xlabel('PFC neuron index')
    ax.set_ylabel('MD neuron index')
    ax.set_title('wPFC2MD '+'PFC learnable-'+str(Elmanlearn)+' Cycle-'+str((i+1)*save_W_step), fontdict=font)
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight')
    fig = ax.get_figure()
    fig.savefig('./animation/'+f'wPFC2MD_index_{i}.png')
    plt.close() # do not show figs in line

images = []
for i in range(len(log['wPFC2MD_list'])):
    filename = './animation/'+f'wPFC2MD_index_{i}.png'
    images.append(imageio.imread(filename))
gif_path = './animation/'+'wPFC2MD_evolution.gif'
imageio.mimsave(gif_path, images, duration=0.2)
optimize(gif_path)


# wMD2PFC evolution
wMD2PFC_min = 0
for i in range(len(log['wMD2PFC_list'])):
    wMD2PFC = log['wMD2PFC_list'][i]
    if  wMD2PFC_min > wMD2PFC.min():
        wMD2PFC_min = wMD2PFC.min()

for i in range(len(log['wMD2PFC_list'])):
    wMD2PFC = log['wMD2PFC_list'][i]
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(wMD2PFC, cmap='Blues_r', vmax=0.0, vmin=wMD2PFC_min) # vmax and vmin need tunning
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_yticks([0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999])
    ax.set_yticklabels([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], rotation=0)
    ax.set_xlabel('MD neuron index')
    ax.set_ylabel('PFC neuron index')
    ax.set_title('wMD2PFC '+'PFC learnable-'+str(Elmanlearn)+' Cycle-'+str((i+1)*save_W_step), fontdict=font)
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight')
    fig = ax.get_figure()
    fig.savefig('./animation/'+f'wMD2PFC_index_{i}.png')
    plt.close() # do not show figs in line

images = []
for i in range(len(log['wMD2PFC_list'])):
    filename = './animation/'+f'wMD2PFC_index_{i}.png'
    images.append(imageio.imread(filename))
gif_path = './animation/'+'wMD2PFC_evolution.gif'
imageio.mimsave(gif_path, images, duration=0.2)
optimize(gif_path)

# MD pretraces evolution
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plot_step = 10

meanMDpreTraces_all = np.mean(MDpreTraces_all, axis=1)
meanMDpreTrace_threshold_all = np.mean(MDpreTrace_threshold_all, axis=1)
for i in range(len(MDpreTraces_all)):
    if (i+1) % plot_step == 0:
        MDpreTraces = meanMDpreTraces_all[i, :]
        MDpreTrace_threshold = meanMDpreTrace_threshold_all[i, :]
        #MDpreTraces = np.convolve(MDpreTraces, np.ones(20)/20, mode='same') # smooth
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
for i in range(len(MDpreTraces_all)):
    if (i+1) % plot_step == 0:
        filename = './animation/'+f'MDpreTraces_index_{i}.png'
        images.append(imageio.imread(filename))
gif_path = './animation/'+'MDpreTraces_evolution.gif'
imageio.mimsave(gif_path, images, duration=0.2)
optimize(gif_path)

# MD outputs evolution
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plot_step = 10

meanMDouts_all = np.mean(MDouts_all, axis=1)
for i in range(len(MDouts_all)):
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
for i in range(len(MDouts_all)):
    if (i+1) % plot_step == 0:
        filename = './animation/'+f'MDoutputs_index_{i}.png'
        images.append(imageio.imread(filename))
gif_path = './animation/'+'MDoutputs_evolution.gif'
imageio.mimsave(gif_path, images, duration=0.2)
optimize(gif_path)