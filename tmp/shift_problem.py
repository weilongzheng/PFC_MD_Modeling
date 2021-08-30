import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
import os
import sys
import pickle
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from task import RikhyeTask
from model_shift import PytorchPFCMD
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from pygifsicle import optimize


config = {
    'RNGSEED': 5,
    'lr': 1e-3,
}

#  set random seed
np.random.seed(config['RNGSEED'])
torch.manual_seed(config['RNGSEED'])

###--------------------------Generate dataset--------------------------###

dataset_config = {
    'Ntrain': 200,      # number of training cycles for each context; default 200
    'Nextra': 200,      # add cycles to show if block1; default 200; if 0, no switch back to past context
    'Ncontexts': 2,     # number of cueing contexts (e.g. auditory cueing context)
    'inpsPerConext': 2, # in a cueing context, there are <inpsPerConext> kinds of stimuli (e.g. auditory cueing context contains high-pass noise and low-pass noise)
    'blockTrain': True,
}
config.update(dataset_config)
         
dataset = RikhyeTask(Ntrain=config['Ntrain'], Nextra=config['Nextra'],
                     Ncontexts=config['Ncontexts'], inpsPerConext=config['inpsPerConext'],
                     blockTrain=config['blockTrain'])

###--------------------------Generate model--------------------------###

model_config = {
    'n_neuron': 1000,
    'n_neuron_per_cue': 200,
    'n_cues': config['Ncontexts']*config['inpsPerConext'],
    'Num_MD': 10,
    'num_active': 5, # num MD active per context
    'n_output': 2,
    'pfcnoise': 1e-3,
    'MDeffect': True,
    'PFClearn': False,
    'shift': 2, # shift step
}
config.update(model_config)

model = PytorchPFCMD(Num_PFC=config['n_neuron'], n_neuron_per_cue=config['n_neuron_per_cue'], n_cues=config['n_cues'],
                     Num_MD=config['Num_MD'], num_active=config['num_active'], num_output=config['n_output'],
                     pfcNoise=config['pfcnoise'], MDeffect=config['MDeffect'])

criterion = nn.MSELoss()

training_params = list()
print("Trainable parameters:")
for name, param in model.named_parameters():
    print(name)
    training_params.append(param)
if config['PFClearn']==True:
    print('pfc.Jrec')
    print('\n', end='')
    training_params.append(model.pfc.Jrec)
else:
    print('\n', end='')
optimizer = torch.optim.Adam(training_params, lr=config['lr'])


tsteps = 200 # time steps in a trial
# total_step = config['Ntrain']*config['Ncontexts'] + config['Nextra']
total_step = config['Ntrain']
print_step = 10 # print statistics every print_step
save_W_step = 1 # save wPFC2MD and wMD2PFC every save_W_step

running_loss = 0.0
running_train_time = 0

log = {
    'stamps': [],
    'mses': [],
    'PFCouts_all': np.zeros(shape=(total_step, tsteps*config['inpsPerConext'], config['n_neuron'])),
    'Jrec_init': model.pfc.Jrec.clone(),
    'Jrec_trained': torch.zeros_like(model.pfc.Jrec.clone()),
}
if config['MDeffect']:
    MD_log = {
        'MDouts_all': np.zeros(shape=(total_step, tsteps*config['inpsPerConext'], config['Num_MD'])),
        'MDpreTraces': np.zeros(shape=(total_step,config['n_neuron'])),
        'MDpreTraces_all': np.zeros(shape=(total_step, tsteps*config['inpsPerConext'], config['n_neuron'])),
        'MDpreTrace_threshold_all': np.zeros(shape=(total_step, tsteps*config['inpsPerConext'], 1)),
        'wPFC2MD_list': [],
        'wMD2PFC_list': [],
    }
    log.update(MD_log)


for i in range(total_step):

    train_time_start = time.time()

    # extract data
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float)
    labels = torch.from_numpy(labels).type(torch.float)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    outputs = model(inputs, labels)

    # save PFC and MD activities
    log['PFCouts_all'][i,:,:] = model.pfc_outputs.detach().numpy()
    if config['MDeffect']:
        log['MDouts_all'][i,:,:] = model.md_output_t
        log['MDpreTraces_all'][i,:,:] = model.md_preTraces
        log['MDpreTrace_threshold_all'][i, :, :] = model.md_preTrace_thresholds

    # backward + optimize
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the norm of gradients 
    if config['PFClearn']:
        torch.nn.utils.clip_grad_norm_(model.pfc.Jrec, 1e-6) # clip the norm of gradients; Jrec 1e-6
    optimizer.step()
    
    #import pdb;pdb.set_trace()

    # shift wIn here
    model.sensory2pfc.shift(shift=config['shift'])
    ###model.md.shift_weights(shift=shift)
    ###print(model.md.wPFC2MD[0, 0:20])
    ###print(model.md.wMD2PFC[0:20, 0])
    ###print(model.sensory2pfc.wIn[:, 0])

    # print statistics
    mse = loss.item()
    log['mses'].append(mse)
    running_loss += mse
    running_train_time += time.time() - train_time_start

    #  save wPFC2MD and wMD2PFC during training
    if i % save_W_step == (save_W_step - 1) and config['MDeffect']:
        log['wPFC2MD_list'].append(model.md.wPFC2MD)
        log['wMD2PFC_list'].append(model.md.wMD2PFC)

    if i % print_step == (print_step - 1):

        print('Total step: {:d}'.format(total_step))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_step, i+1))
        print('shift: {:d}'.format(config['shift']))

        # running loss
        print('loss: {:0.5f}'.format(running_loss / print_step))
        running_loss = 0.0

        # training time
        print('Predicted left training time: {:0.0f} s'.format(
            (running_train_time) * (total_step - i - 1) / print_step),
            end='\n\n')
        running_train_time = 0

        # if savemodel:
        #     # save model every print_step
        #     fname = os.path.join('models', model_name + '.pt')
        #     torch.save(model.state_dict(), fname)
        #     # save info of the model
        #     fpath = os.path.join('models', model_name + '.txt')
        #     with open(fpath, 'w') as f:
        #         f.write('input_size = ' + str(input_size) + '\n')
        #         f.write('hidden_size = ' + str(hidden_size) + '\n')
        #         f.write('output_size = ' + str(output_size) + '\n')
        #         f.write('num_layers = ' + str(num_layers) + '\n')


# Save weights after training
log['Jrec_trained'] = model.pfc.Jrec

# Save model
# filename = Path('files')
# os.makedirs(filename, exist_ok=True)
# file_training = 'Animation'+'Ntrain'+str(Ntrain)+'_Nextra'+str(Nextra)+'_train_numMD'+str(Num_MD)+\
#                 '_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+\
#                 '_shift'+str(shift)+'_R'+str(RNGSEED)+'.pkl'
# with open(filename / file_training, 'wb') as f:
#     pickle.dump(log, f)

print('Finished Training')


###--------------------------Analysis--------------------------###

# Make some plots

# Plot MSE curve
plt.plot(log['mses'])
plt.xlabel('Cycles')
plt.ylabel('MSE loss')
#plt.xticks([0, 500, 1000, 1200])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
plt.show()

# Heatmap wPFC2MD
ax = plt.figure(figsize=(8, 6))
ax = sns.heatmap(model.md.wPFC2MD, cmap='Reds')
ax.set_xticks([0, 999])
ax.set_xticklabels([1, 1000], rotation=0)
ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_xlabel('PFC neuron index')
ax.set_ylabel('MD neuron index')
ax.set_title('wPFC2MD')
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight')
plt.tight_layout()
plt.show()

# Heatmap wMD2PFC
ax = plt.figure(figsize=(8, 6))
ax = sns.heatmap(model.md.wMD2PFC, cmap='Blues_r')
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
ax.set_yticks([0, 999])
ax.set_yticklabels([1, 1000], rotation=0)
ax.set_xlabel('MD neuron index')
ax.set_ylabel('PFC neuron index')
ax.set_title('wMD2PFC')
cbar = ax.collections[0].colorbar
cbar.set_label('connection weight')
plt.tight_layout()
plt.show()


# wPFC2MD evolution
font = {'family':'Times New Roman','weight':'normal', 'size':15}
wPFC2MD_max = 0
for i in range(len(log['wPFC2MD_list'])):
    wPFC2MD = log['wPFC2MD_list'][i]
    if  wPFC2MD_max < wPFC2MD.max():
        wPFC2MD_max = wPFC2MD.max()

for i in range(len(log['wPFC2MD_list'])):
    wPFC2MD = log['wPFC2MD_list'][i]
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(wPFC2MD, cmap='Reds', vmax=wPFC2MD_max, vmin=0.0)
    ax.set_xticks([0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999])
    ax.set_xticklabels([1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], rotation=0)
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_xlabel('PFC neuron index')
    ax.set_ylabel('MD neuron index')
    ax.set_title('wPFC2MD '+' Cycle-'+str((i+1)*save_W_step), fontdict=font)
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


## plot pfc2md weights
# wPFC2MD = log['wPFC2MD']
# number = Num_MD
# cmap = plt.get_cmap('rainbow') 
# colors = [cmap(i) for i in np.linspace(0,1,number)]
# plt.figure(figsize=(18,20))
# for i,color in enumerate(colors, start=1):
#     plt.subplot(number,1,i)
#     plt.plot(wPFC2MD[i-1,:],color=color)
# plt.xlabel(f'wPFC2MD; shift step = {shift}')
# #plt.suptitle(f'wPFC2MD; shift step = {shift}')
# plt.show()

## plot md2pfc weights
# wMD2PFC = log['wMD2PFC']
# number = Num_MD
# cmap = plt.get_cmap('rainbow') 
# colors = [cmap(i) for i in np.linspace(0,1,number)]
# plt.figure(figsize=(18, 20))
# for i,color in enumerate(colors, start=1):
#     plt.subplot(number,1,i)
#     plt.plot(wMD2PFC[:,i-1],color=color)
# plt.xlabel(f'wMD2PFC; shift step = {shift}')
# #plt.suptitle(f'wMD2PFC; shift step = {shift}')
# plt.show()

## plot pfc recurrent weights before and after training
#fig, axes = plt.subplots(nrows=1, ncols=2)
#Jrec = model.pfc.Jrec.detach().numpy()
#Jrec_init = Jrec_init
## find minimum of minima & maximum of maxima
#minmin = np.min([np.min(Jrec_init), np.min(Jrec)])
#maxmax = np.max([np.max(Jrec_init), np.max(Jrec)])
#num_show = 200
#im1 = axes[0].imshow(Jrec_init[:num_show,:num_show], vmin=minmin, vmax=maxmax,
#                     extent=(0,num_show,0,num_show), cmap='viridis')
#im2 = axes[1].imshow(Jrec[:num_show,:num_show], vmin=minmin, vmax=maxmax,
#                     extent=(0,num_show,0,num_show), cmap='viridis')
#
## add space for colour bar
#fig.subplots_adjust(right=0.85)
#cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
#fig.colorbar(im2, cax=cbar_ax)
