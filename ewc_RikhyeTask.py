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

dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule,\
                          rule=rule, blocklen=blocklen, block_cueingcontext=block_cueingcontext,\
                          tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)


total_step = sum(blocklen)//batch_size

tsteps = 200
log = defaultdict(list)

#import pdb;pdb.set_trace()
for iblock in range(3):
    print('Training {:d} block'.format(iblock))
    inputs_all = torch.zeros((blocklen[iblock],400,1,4))
    outputs_all = torch.zeros((blocklen[iblock],400,1,2))
    
    for i in range(blocklen[iblock]):
    
        train_time_start = time.time()
        
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
        log['mse'].append(mse)
    
    dataset_ewc = TrainingDataset(inputs_all,outputs_all)
    ewc.register_ewc_params(dataset_ewc, 1, blocklen[iblock])
        
# Plot MSE curve
plt.plot(log['mse'], label='With MD')
plt.xlabel('Cycles')
plt.ylabel('MSE loss')
plt.legend()
plt.tight_layout()
plt.show()        
        
        
        