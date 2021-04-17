'''
Elman with MD layer
'''

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')

from task import RikhyeTaskBatch
from model import Elman_MD


#---------------- Helper funtions ----------------#
def disjoint_penalty(model, reg=1e-4):
    '''
    Keep weight matrices disjoint by adding ||matmul(W.T, W)||1 to loss
    '''
    norm = torch.tensor(0.)
    # for name, param in model.named_parameters():
    #     if 'rnn.input2h.weight' in name or 'rnn.h2h.weight' in name:
    #         norm += reg * torch.abs(torch.matmul(param.t(), param)).sum()
    Winput2h = model.parm['rnn.input2h.weight']
    Wrec = model.parm['rnn.h2h.weight']
    for param in [Winput2h, Wrec]:
        norm += reg * torch.abs(torch.matmul(param.t(), param)).sum()
    return norm


#---------------- Rikhye dataset with batch dimension ----------------#

# set random seed
RNGSEED = 5
np.random.seed([RNGSEED])
torch.manual_seed(RNGSEED) 
os.environ['PYTHONHASHSEED'] = str(RNGSEED)

num_cueingcontext = 2
num_cue = 2
num_rule = 2
rule = [0, 1, 0, 1]
blocklen = [200, 200, 200]
block_cueingcontext = [0, 1, 0]
tsteps = 200
cuesteps = 100
batch_size = 1


# save dataset settings

log = dict()

dataset_config = dict()
dataset_config['RNGSEED'] = RNGSEED
dataset_config['num_cueingcontext'] = num_cueingcontext
dataset_config['num_cue'] = num_cue
dataset_config['num_rule'] = num_rule
dataset_config['rule'] = rule
dataset_config['blocklen'] = blocklen
dataset_config['block_cueingcontext'] = block_cueingcontext
dataset_config['tsteps'] = tsteps
dataset_config['cuesteps'] = cuesteps
dataset_config['batch_size'] = batch_size

log['dataset_config'] = dataset_config


# create a dataset
dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule,\
                          rule=rule, blocklen=blocklen, block_cueingcontext=block_cueingcontext,\
                          tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)


#---------------- Elman_MD model ----------------#
testmodel = True # set False if train & save models

input_size = 4          # 4 cues
hidden_size = 1000      # number of PFC neurons
output_size = 2         # 2 rules
num_layers = 1
nonlinearity = 'tanh'
Num_MD = 10
num_active = 5
reg = 1e-5              # disjoint penalty regularization
MDeffect = True
Sensoryinputlearn = True
Elmanlearn = True


# save model settings
model_config = dict()
model_config['input_size'] = input_size
model_config['hidden_size'] = hidden_size
model_config['output_size'] = output_size
model_config['num_layers'] = num_layers
model_config['nonlinearity'] = nonlinearity
model_config['Num_MD'] = Num_MD
model_config['num_active'] = num_active
model_config['reg'] = reg
model_config['MDeffect'] = MDeffect
model_config['Sensoryinputlearn'] = Sensoryinputlearn
model_config['Elmanlearn'] = Elmanlearn

log['model_config'] = model_config

# create model
model = Elman_MD(input_size=input_size, hidden_size=hidden_size, output_size=output_size,\
                 num_layers=num_layers, nonlinearity=nonlinearity, Num_MD=Num_MD, num_active=num_active,\
                 tsteps=tsteps, MDeffect=MDeffect)

# print model
print('MDeffect:', MDeffect, end='\n\n')
print(model, end='\n\n')

# create directory for saving models
model_name = 'Elman_MD'+'_MDeffect'+str(MDeffect)+'_Sensoryinputlearn'+str(Sensoryinputlearn)+\
             '_Elmanlearn'+str(Elmanlearn)+'_R'+str(RNGSEED)
if testmodel:
    model_name = 'test_' + model_name
    directory = Path('temp_files')
    os.makedirs(directory, exist_ok=True)
else:
    directory = Path('files')
    os.makedirs(directory, exist_ok=True)


#---------------- Training ----------------#

# set training parameters
training_params = list()
print("Trainable parameters:")
for name, param in model.named_parameters():
    if Sensoryinputlearn == False and 'rnn.input2h' in name:
        continue
    if Elmanlearn == False and 'rnn.h2h' in name:
        continue
    print(name, param.shape)
    training_params.append(param)
print('\n', end='')
optimizer = torch.optim.Adam(training_params, lr=1e-3)

criterion = nn.MSELoss()

total_step = sum(blocklen)//batch_size
print_step = 10
running_loss = 0.0
running_mseloss = 0.0
running_train_time = 0

log['loss_val'] = []
log['mse'] = []


for i in range(total_step):

    train_time_start = time.time()

    # extract data
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float)
    labels = torch.from_numpy(labels).type(torch.float)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs, labels)
    ####print('MSE', criterion(outputs, labels))
    ####print('reg', disjoint_penalty(model, reg=reg))

    loss = criterion(outputs, labels) + disjoint_penalty(model, reg=reg)
    ####print(loss)
    ####print(model.parm['rnn.input2h.weight'])
    ####print(model.parm['rnn.h2h.weight'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients
    optimizer.step()

    # save loss values
    mse = criterion(outputs, labels).item()
    loss_val = loss.item()
    log['mse'].append(mse)
    log['loss_val'].append(loss_val)

    # print statistics
    running_train_time += time.time() - train_time_start
    running_loss += loss_val
    running_mseloss += mse
    if i % print_step == (print_step - 1):

        print('Total step: {:d}'.format(total_step))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_step, i+1))

        # running loss
        print('Total loss: {:0.5f};'.format(running_loss / print_step), 'MSE loss: {:0.5f}'.format(running_mseloss / print_step))
        running_loss = 0.0
        running_mseloss = 0.0

        # training time
        print('Predicted left training time: {:0.0f} s'.format(
        (running_train_time) * (total_step - i - 1) / print_step),
        end='\n\n')
        running_train_time = 0

        # save model during training
        log['Winput2h'] = model.parm['rnn.input2h.weight'].data.detach().numpy()
        log['Wrec'] = model.parm['rnn.h2h.weight'].data.detach().numpy()
        if MDeffect == True:  
            log['wPFC2MD'] = model.md.wPFC2MD
            log['wMD2PFC'] = model.md.wMD2PFC
            log['wMD2PFCMult'] = model.md.wMD2PFCMult

        with open(directory / (model_name + '.pkl'), 'wb') as f:
            pickle.dump(log, f)
        torch.save(model.state_dict(), directory / (model_name + '.pth'))

print('Finished Training')


#---------------- Make some plots ----------------#

with open(directory / (model_name + '.pkl'), 'rb') as f:
    log = pickle.load(f)

font = {'family':'Times New Roman','weight':'normal', 'size':24}

# Plot loss curve
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


# Plot connection weights
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
im = plt.matshow(Wrec, cmap='bwr')
plt.xlabel('Elman neuron index', fontdict=font)
plt.ylabel('Elman neuron index', fontdict=font)
plt.xticks(ticks=[0, 999], labels=[1, 1000])
plt.yticks(ticks=[0, 999], labels=[1, 1000])
plt.title('Weights: recurrent', fontdict=font)
plt.colorbar(im)
plt.show()