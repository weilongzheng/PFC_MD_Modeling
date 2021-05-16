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
from model import PytorchPFCMD
import matplotlib.pyplot as plt
import seaborn as sns

# Generate trainset
RNGSEED = 3 # set random seed
np.random.seed([RNGSEED])
torch.manual_seed(RNGSEED)

Ntrain = 100            # number of training cycles for each context
Nextra = 100            # add cycles to show if block1
Ncontexts = 2           # number of cueing contexts (e.g. auditory cueing context)
inpsPerConext = 2       # in a cueing context, there are <inpsPerConext> kinds of stimuli
                         # (e.g. auditory cueing context contains high-pass noise and low-pass noise)
dataset = RikhyeTask(Ntrain=Ntrain, Nextra=Nextra, Ncontexts=Ncontexts, inpsPerConext=inpsPerConext, blockTrain=True)

# Model settings
n_neuron = 1000
n_neuron_per_cue = 200
Num_MD = 10
num_active = 5  # num MD active per context
n_output = 2
noiseSD = 1e-1
MDeffect = True
PFClearn = False
noiseInput = False # additional noise input neuron 

model = PytorchPFCMD(Num_PFC=n_neuron, n_neuron_per_cue=n_neuron_per_cue, Num_MD=Num_MD, num_active=num_active, num_output=n_output, \
MDeffect=MDeffect, noiseInput = noiseInput)

# Training
criterion = nn.MSELoss() 
training_params = list()
for name, param in model.named_parameters():
    print(name)
    training_params.append(param)

if PFClearn==True:
    print('pfc.Jrec')
    print('\n')
    training_params.append(model.pfc.Jrec)
    
Jrec_init = model.pfc.Jrec.clone()#.numpy()
print(Jrec_init)
optimizer = torch.optim.Adam(training_params, lr=1e-3)
#import pdb;pdb.set_trace()

total_step = Ntrain*Ncontexts+Nextra
print_step = 10
running_loss = 0.0
running_train_time = 0
mses = list()
losses = []
timestamps = []
model_name = 'model-' + str(int(time.time()))
savemodel = False
log = defaultdict(list)
MDpreTraces = np.zeros(shape=(total_step,n_neuron))
#MDouts_all = np.zeros(shape=(total_step,Num_MD))
#PFCouts_all = np.zeros(shape=(total_step,n_neuron))
tsteps = 200
MDouts_all = np.zeros(shape=(total_step*inpsPerConext,tsteps,Num_MD))
PFCouts_all = np.zeros(shape=(total_step*inpsPerConext,tsteps,n_neuron))
outputs_all = np.zeros(shape=(total_step*inpsPerConext,tsteps,2))
for i in range(total_step):

    train_time_start = time.time()

    # extract data
    inputs, labels = dataset()
    if noiseInput == True:
        inputs = np.hstack((inputs,np.random.normal(size=(inputs.shape[0],1)) * noiseSD))

    #import pdb;pdb.set_trace()    
    inputs = torch.from_numpy(inputs).type(torch.float)
    labels = torch.from_numpy(labels).type(torch.float)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs, labels)
    #PFCouts_all[i,:] = model.pfc.activity.detach().numpy()
#    if  MDeffect == True:
#        MDouts_all[i,:] = model.md_output
#        MDpreTraces[i,:] = model.md.MDpreTrace
    tstart = 0
    for itrial in range(inpsPerConext): 
        PFCouts_all[i*inpsPerConext+tstart,:,:] = model.pfc_outputs.detach().numpy()[tstart*tsteps:(tstart+1)*tsteps,:]
        outputs_all[i*inpsPerConext+tstart,:,:] = outputs.detach().numpy()[tstart*tsteps:(tstart+1)*tsteps,:]
        if  MDeffect == True:
            MDouts_all[i*inpsPerConext+tstart,:,:] = model.md_output_t[tstart*tsteps:(tstart+1)*tsteps,:]
        tstart += 1

    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # normalization  
    if PFClearn==True:
        torch.nn.utils.clip_grad_norm_(model.pfc.Jrec, 1e-6) # normalization Jrec 1e-6
    optimizer.step()
    
    #
    # print statistics
    mse = loss.item()
    log['mse'].append(mse)
    running_train_time += time.time() - train_time_start
    running_loss += loss.item()

    if i % print_step == (print_step - 1):

        print('Total step: {:d}'.format(total_step))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_step, i+1))

        # running loss
        print('loss: {:0.5f}'.format(running_loss / print_step))
        losses.append(running_loss / print_step)
        timestamps.append(i+1-print_step)
        running_loss = 0.0

        # training time
        print('Predicted left training time: {:0.0f} s'.format(
        (running_train_time) * (total_step - i - 1) / print_step),
        end='\n\n')
        running_train_time = 0
        print(model.pfc.Jrec)
        if savemodel:
            # save model every print_step
            fname = os.path.join('models', model_name + '.pt')
            torch.save(model.state_dict(), fname)

            # save info of the model
            fpath = os.path.join('models', model_name + '.txt')
            with open(fpath, 'w') as f:
                f.write('input_size = ' + str(input_size) + '\n')
                f.write('hidden_size = ' + str(hidden_size) + '\n')
                f.write('output_size = ' + str(output_size) + '\n')
                f.write('num_layers = ' + str(num_layers) + '\n')

print('Finished Training')


if  MDeffect == True:  
    log['wPFC2MD'] = model.md.wPFC2MD
    log['wMD2PFC'] = model.md.wMD2PFC
    log['wMD2PFCMult'] = model.md.wMD2PFCMult

filename = Path('files')
os.makedirs(filename, exist_ok=True)
file_training = 'train_softSwitch10_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+'_R'+str(RNGSEED)+'.pkl'
with open(filename / file_training, 'wb') as f:
    pickle.dump(log, f)
    
# Plot MSE curve
plt.plot(log['mse'], label='With MD')
plt.xlabel('Cycles')
plt.ylabel('MSE loss')
plt.legend()
#plt.xticks([0, 500, 1000, 1200])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
plt.show()

## plot pfc2md and md2pfc weights
if  MDeffect == True: 
    ## plot pfc2md weights
    wPFC2MD = log['wPFC2MD']
    wMD2PFC = log['wMD2PFC']
    ax = plt.figure()
    ax = sns.heatmap(wPFC2MD, cmap='Reds')
    ax.set_xticks([0, 999])
    ax.set_xticklabels([1, 1000], rotation=0)
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_xlabel('PFC neuron index')
    ax.set_ylabel('MD neuron index')
    ax.set_title('wPFC2MD '+'PFC learnable-'+str(PFClearn))
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight')
    plt.tight_layout()
    plt.show()
    
    # Heatmap wMD2PFC
    ax = plt.figure()
    ax = sns.heatmap(wMD2PFC, cmap='Blues_r')
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_yticks([0, 999])
    ax.set_yticklabels([1, 1000], rotation=0)
    ax.set_xlabel('MD neuron index')
    ax.set_ylabel('PFC neuron index')
    ax.set_title('wMD2PFC '+'PFC learnable-'+str(PFClearn))
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight')
    plt.tight_layout()
    plt.show()
    
    # wPFC2MD = log['wPFC2MD']
    # number = Num_MD
    # cmap = plt.get_cmap('rainbow') 
    # colors = [cmap(i) for i in np.linspace(0,1,number)]
    # plt.figure()
    # for i,color in enumerate(colors, start=1):
    #     plt.subplot(number,1,i)
    #     plt.plot(wPFC2MD[i-1,:],color=color)
    # plt.suptitle('wPFC2MD')
    
    # wMD2PFC = log['wMD2PFC']
    # number = Num_MD
    # cmap = plt.get_cmap('rainbow') 
    # colors = [cmap(i) for i in np.linspace(0,1,number)]
    # plt.figure()
    # for i,color in enumerate(colors, start=1):
    #     plt.subplot(number,1,i)
    #     plt.plot(wMD2PFC[:,i-1],color=color)
    # plt.suptitle('wMD2PFC')

## plot pfc recurrent weights before and after training
Jrec = model.pfc.Jrec.detach().numpy()
Jrec_init = Jrec_init.detach().numpy()
fig, axes = plt.subplots(nrows=1, ncols=2)
# find minimum of minima & maximum of maxima
minmin = np.min([np.percentile(Jrec_init,10), np.percentile(Jrec,10)])
maxmax = np.max([np.percentile(Jrec_init,90), np.percentile(Jrec,90)])
num_show = 200
im1 = axes[0].imshow(Jrec_init[:num_show,:num_show], vmin=minmin, vmax=maxmax,
                     extent=(0,num_show,0,num_show), cmap='viridis')
im2 = axes[1].imshow(Jrec[:num_show,:num_show], vmin=minmin, vmax=maxmax,
                     extent=(0,num_show,0,num_show), cmap='viridis')

# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

