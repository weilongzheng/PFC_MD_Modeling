import torch
import torch.nn as nn
import torch.nn.functional as F
from elastic_weight_consolidation import ElasticWeightConsolidation
from task import RikhyeTask
from task import RikhyeTaskBatch
import time
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import matplotlib as mpl

class Elman(nn.Module):
    
    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        if nonlinearity == 'relu':
            self.activation = torch.relu
        else:
            self.activation = torch.tanh
            
        self.input2h = nn.Linear(input_size, hidden_size)
        k = (1./self.hidden_size)**0.5
        nn.init.uniform_(self.input2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        nn.init.uniform_(self.input2h.bias, a=-k, b=k)

        self.h2h = nn.Linear(hidden_size, hidden_size)
        k = (1./self.hidden_size)**0.5
        nn.init.uniform_(self.h2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        nn.init.uniform_(self.h2h.bias, a=-k, b=k)

        
    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, input, hidden=None, mdinput=None):

        if hidden is None:
            hidden = self.init_hidden(input)

        pre_activation = self.input2h(input) + self.h2h(hidden)

        if mdinput is not None:
            pre_activation += mdinput
        
        hidden = self.activation(pre_activation)
        
        return hidden
    
class Elman_model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, nonlinearity):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = Elman(input_size, hidden_size, nonlinearity)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, input):

        n_time = input.shape[0]
        batch_size = input.shape[1]

        RNN_output = torch.zeros((n_time, batch_size, self.hidden_size))
        RNN_hidden_t = torch.zeros((batch_size, self.hidden_size))
        
        for t in range(n_time):
            input_t = input[t, ...].unsqueeze(dim=0)

            RNN_hidden_t = self.rnn(input_t, RNN_hidden_t)
            RNN_output[t, :, :] = RNN_hidden_t

        model_out = self.fc(RNN_output)
        model_out = torch.tanh(model_out)

        return model_out
    
class TrainingDataset(Dataset):
    
    def __init__(self, input, output):
            self.output = output
            self.input = input
            
    def __len__(self):
            return len(self.output)
        
    def __getitem__(self, idx):
            output = self.output[idx]
            input = self.input[idx]
            return input,output
    
#Ntrain = 50
#Nextra = 0 
#Ncontexts = 2
#inpsPerConext = 2

input_size = 4          # 4 cues
hidden_size = 256      # number of PFC neurons
output_size = 2         # 2 rules
nonlinearity = 'tanh'
model = Elman_model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, nonlinearity=nonlinearity)
crit = nn.MSELoss()
ewc = ElasticWeightConsolidation(model, crit=crit, lr=1e-4)

num_cueingcontext = 2
num_cue = 2
num_rule = 2
rule = [0, 1, 0, 1]
blocklen = [200, 200, 100]
block_cueingcontext = [0, 1, 0]
tsteps = 200
cuesteps = 100
batch_size = 1
total_step = sum(blocklen)//batch_size
tsteps = 200

dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule,\
                          rule=rule, blocklen=blocklen, block_cueingcontext=block_cueingcontext,\
                          tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)

log_ewc = defaultdict(list)
#import pdb;pdb.set_trace()
for iblock in range(len(blocklen)):
    print('Training {:d} block with EWC'.format(iblock))
    inputs_all = torch.zeros((blocklen[iblock],400,1,4))
    outputs_all = torch.zeros((blocklen[iblock],400,1,2))
    
    for i in range(blocklen[iblock]):
    
        # extract data
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float)
        labels = torch.from_numpy(labels).type(torch.float)
        inputs_all[i,:] = inputs
        outputs_all[i,:] = labels
        
        ewc.forward_backward_update(inputs, labels)
        
        outputs = ewc.model(inputs)    
        loss = crit(outputs, labels)
        mse = loss.item()
        log_ewc['mse'].append(mse)
    
    dataset_ewc = TrainingDataset(inputs_all,outputs_all)
    ewc.register_ewc_params(dataset_ewc, 1, blocklen[iblock])

## Tese Elman RNN wihout EWC
dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule,\
                          rule=rule, blocklen=blocklen, block_cueingcontext=block_cueingcontext,\
                          tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)
    
model = Elman_model(input_size=input_size, hidden_size=hidden_size, output_size=output_size, nonlinearity=nonlinearity)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

log = defaultdict(list)
for i in range(total_step):
    # extract data
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float)
    labels = torch.from_numpy(labels).type(torch.float)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    outputs = model(inputs)
    
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients
    optimizer.step()

    # save loss values
    mse = criterion(outputs, labels).item()
    loss_val = loss.item()
    log['mse'].append(mse)

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
        
# Plot MSE curve
filesave = Path('results')
os.makedirs(filesave, exist_ok=True)

plt.figure(figsize=(2.4,2.4))
plt.plot(log_ewc['mse'], 'tab:blue', label='Elman RNN with EWC')
plt.plot(log['mse'], 'tab:red', label='Elman RNN without EWC')
plt.xticks(np.arange(0,501,100),np.arange(0,1001,200))
plt.xlabel('Trials'),plt.ylabel('MSE')
plt.ylim(0, 0.6)
plt.legend(frameon=False)
plt.axvspan(0, 200, ymin=0, ymax=1, alpha=0.1, color='tab:orange')
plt.axvspan(200, 400, ymin=0, ymax=1, alpha=0.1, color='tab:orange')
plt.axvspan(400, 500, ymin=0, ymax=1, alpha=0.1, color='tab:green')
plt.title('Model Performance')
plt.tight_layout()
plt.savefig(filesave/'mse_ewc.pdf', dpi=300) 
        
        
        