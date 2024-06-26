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
import imageio
from pygifsicle import optimize



# Generate trainset
RNGSEED = 5 # set random seed
np.random.seed([RNGSEED])

Ntrain = 500            # number of training cycles for each context; default 200
Nextra = 0            # add cycles to show if block1; default 200
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
save_W_step = 20

# Get data
filename = Path('files')
os.makedirs(filename, exist_ok=True)
# file_training = 'train_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+\
#                 '_PFC'+str(PFClearn)+'_shift'+str(shift)+'_R'+str(RNGSEED)+'.pkl'
file_training = 'AnimationNtrain'+str(Ntrain)+'_Nextra'+str(Nextra)+'_train_numMD'+str(Num_MD)+\
                '_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+\
                '_shift'+str(shift)+'_R'+str(RNGSEED)+'.pkl'
with open(filename / file_training, 'rb') as f:
    log = pickle.load(f)

wPFC2MD = log['wPFC2MD']
wMD2PFC = log['wMD2PFC']
# Jrec = log['Jrec']
# Jrec = Jrec.detach().numpy()


# Plot MSE curve
plt.figure(),plt.plot(log['mse'], label=f'With MD; shift step = {shift}')
plt.xlabel('Cycles')
plt.ylabel('MSE loss')
plt.legend()
#plt.xticks([0, 500, 1000, 1200])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
plt.show()

# Heatmap wPFC2MD
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(wPFC2MD, cmap='Reds')
ax.set_xticks([0, 999])
ax.set_xticklabels([1, 1000], rotation=0)
ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_xlabel('PFC neuron index')
ax.set_ylabel('MD neuron index')
ax.set_title('wPFC2MD '+'PFC learnable-'+str(PFClearn)+' Shift-'+str(shift))
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight')
plt.tight_layout()
plt.show()

# Heatmap wMD2PFC
ax = plt.figure(figsize=(15, 10))
ax = sns.heatmap(wMD2PFC, cmap='Blues_r')
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_yticks([0, 999])
ax.set_yticklabels([1, 1000], rotation=0)
ax.set_xlabel('MD neuron index')
ax.set_ylabel('PFC neuron index')
ax.set_title('wMD2PFC '+'PFC learnable-'+str(PFClearn)+' Shift-'+str(shift))
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight')
plt.tight_layout()
plt.show()

# Heatmap Jrec
## the Jrec is too large. if use seaborn, the kernel would fail.
# ax = plt.figure(figsize=(15, 15))
# ax = sns.heatmap(Jrec, cmap='bwr')
# ax.set_xticks([0, 999])
# ax.set_xticklabels([1, 1000], rotation=0)
# ax.set_yticks([0, 999])
# ax.set_yticklabels([1, 1000], rotation=0)
# ax.set_xlabel('PFC neuron index')
# ax.set_ylabel('PFC neuron index')
# ax.set_title('Jrec '+'PFC learnable-'+str(PFClearn)+' shift step-'+str(shift))
# cbar = ax.collections[0].colorbar
# cbar.set_label('connection weight')

# plt.figure(figsize=(15, 15))
# plt.matshow(Jrec, cmap='bwr')
# plt.xlabel('PFC neuron index')
# plt.ylabel('PFC neuron index')
# plt.xticks(ticks=[0, 999], labels=[1, 1000])
# plt.yticks(ticks=[0, 999], labels=[1, 1000])
# plt.title('Jrec '+'PFC learnable-'+str(PFClearn)+' shift step-'+str(shift))
# plt.show()



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
    ax.set_title('wPFC2MD '+'PFC learnable-'+str(PFClearn)+' Shift-'+str(shift)+' Cycle-'+str((i+1)*save_W_step), fontdict=font)
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
    ax.set_title('wMD2PFC '+'PFC learnable-'+str(PFClearn)+' Shift-'+str(shift)+' Cycle-'+str((i+1)*save_W_step), fontdict=font)
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

# PFC outputs evolution
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plot_step = 10

meanPFCouts_all = np.mean(PFCouts_all, axis=1)
for i in range(len(PFCouts_all)):
    if (i+1) % plot_step == 0:
        meanPFCouts = meanPFCouts_all[i, :]
        plt.plot(meanPFCouts)
        plt.title('PFC outputs' + ' Cycle-'+str(i+1), fontdict=font)
        plt.xlabel('PFC neuron index')
        plt.ylabel('PFC activities')
        plt.xlim([-50, 1050])
        plt.ylim([0.0, 1.0])
        plt.xticks(ticks=[100*i for i in range(11)], rotation=0)
        plt.yticks(ticks=[0.1*i for i in range(11)], rotation=0)
        plt.savefig('./animation/'+f'PFCoutputs_index_{i}.png')
        plt.close() # do not show figs in line

images = []
for i in range(len(PFCouts_all)):
    if (i+1) % plot_step == 0:
        filename = './animation/'+f'PFCoutputs_index_{i}.png'
        images.append(imageio.imread(filename))
gif_path = './animation/'+'PFCoutputs_evolution.gif'
imageio.mimsave(gif_path, images, duration=0.2)
optimize(gif_path)