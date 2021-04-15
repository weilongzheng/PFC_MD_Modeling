'''
Pytorch built-in Elman RNN + MD layer
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

from temp_task import RikhyeTaskBatch
from temp_model import Elman_MD


log = dict()

#---------------- Rikhye dataset with batch dimension ----------------#
RNGSEED = 5
np.random.seed([RNGSEED])
os.environ['PYTHONHASHSEED'] = str(RNGSEED)

num_cueingcontext = 2
num_cue = 2
num_rule = 2
rule = [0, 1, 0, 1]
#blocklen = [500, 500, 200]
blocklen = [10, 10, 10]
block_cueingcontext = [0, 1, 0]
tsteps = 200
cuesteps = 100
batch_size = 1


# save dataset settings
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
input_size = 4 # 4 cues
hidden_size = 200 # number of PFC neurons
output_size = 2 # 2 rules
num_layers = 1
nonlinearity = 'tanh'
Num_MD = 10
num_active = 5
MDeffect = True
Sensoryinputlearn = False
Elmanlearn = False


# save model settings
model_config = dict()
model_config['input_size'] = input_size
model_config['hidden_size'] = hidden_size
model_config['output_size'] = output_size
model_config['num_layers'] = num_layers
model_config['nonlinearity'] = nonlinearity
model_config['Num_MD'] = Num_MD
model_config['num_active'] = num_active
model_config['MDeffect'] = MDeffect
model_config['Sensoryinputlearn'] = Sensoryinputlearn
model_config['Elmanlearn'] = Elmanlearn

log['model_config'] = model_config

# create a model
model = Elman_MD(input_size=input_size, hidden_size=hidden_size, output_size=output_size,\
                 num_layers=num_layers, nonlinearity=nonlinearity, Num_MD=Num_MD, num_active=num_active,\
                 tsteps=tsteps, MDeffect=MDeffect)
print(model)

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
running_train_time = 0
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
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # normalization
    optimizer.step()

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
        running_loss = 0.0

        # training time
        print('Predicted left training time: {:0.0f} s'.format(
        (running_train_time) * (total_step - i - 1) / print_step),
        end='\n\n')
        running_train_time = 0

        # save model during training
        if  MDeffect == True:  
            log['wPFC2MD'] = model.md.wPFC2MD
            log['wMD2PFC'] = model.md.wMD2PFC
            log['wMD2PFCMult'] = model.md.wMD2PFCMult

        directory = Path('files')
        os.makedirs(directory, exist_ok=True)
        model_name = 'Elman_MD'+'_MDeffect'+str(MDeffect)+'_Sensoryinputlearn'+str(Sensoryinputlearn)+\
                     '_Elmanlearn'+str(Elmanlearn)+'_R'+str(RNGSEED)
        with open(directory / (model_name + '.pkl'), 'wb') as f:
            pickle.dump(log, f)
        torch.save(model.state_dict(), directory / (model_name + '.pth'))

print('Finished Training')



#---------------- Make some plots ----------------#
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