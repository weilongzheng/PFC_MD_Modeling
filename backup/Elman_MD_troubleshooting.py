import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from pygifsicle import optimize
import pickle
from pathlib import Path
import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')

from task import RikhyeTaskBatch


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
blocklen = [200, 200, 100]
block_cueingcontext = [0, 1, 0]
tsteps = 200
cuesteps = 100
batch_size = 1


# create a dataset
dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule,\
                          rule=rule, blocklen=blocklen, block_cueingcontext=block_cueingcontext,\
                          tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)

class Elman(nn.Module):
    """Elman RNN that can take in MD inputs.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
    Inputs:
        input: (seq_len, batch, input_size), external input; seq_len is set to 1
        hidden: (batch, hidden_size), initial hidden activity;
        mdinput: (batch, hidden_size), MD input;

    Acknowlegement:
        based on Robert Yang's CTRNN class
    """


    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        if nonlinearity == 'relu':
            self.activation = torch.relu
        else:
            self.activation = torch.tanh

        # Sensory input -> RNN
        self.input2h = nn.Linear(input_size, hidden_size)
        k = (1./self.hidden_size)**0.5
        nn.init.uniform_(self.input2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        nn.init.uniform_(self.input2h.bias, a=-k, b=k)

        # RNN -> RNN
        self.h2h = nn.Linear(hidden_size, hidden_size)
        k = (1./self.hidden_size)**0.5
        nn.init.uniform_(self.h2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        nn.init.uniform_(self.h2h.bias, a=-k, b=k)

        
    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, input, hidden=None, mdinput=None):
        '''
        Propogate input through the network
        '''
        # TODO: input.shape has to be [timestep=1, batch_size, input_size]
        
        if hidden is None:
            hidden = self.init_hidden(input)

        pre_activation = self.input2h(input) + self.h2h(hidden)

        if mdinput is not None:
            pre_activation += mdinput
        
        hidden = self.activation(pre_activation)
        
        return hidden


class Elman_MD(nn.Module):
    """Elman RNN with a MD layer
    Parameters:
    input_size: int, RNN input size
    hidden_size: int, RNN hidden size
    output_size: int, output layer size
    num_layers: int, number of RNN layers
    nonlinearity: str, 'tanh' or 'relu', nonlinearity in RNN layers
    Num_MD: int, number of neurons in MD layer
    num_active: int, number of active neurons in MD layer (refer to top K winner-take-all)
    tsteps: int, length of a trial, equals to cuesteps + delaysteps
    """


    def __init__(self, input_size, hidden_size, output_size, num_layers, nonlinearity, Num_MD, num_active, tsteps, MDeffect=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.tsteps = tsteps

        # PFC layer / Elman RNN layer
        self.rnn = Elman(input_size, hidden_size, nonlinearity)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, target):
        '''
        Propogate input through the network
        '''
        n_time = input.shape[0]
        batch_size = input.shape[1]

        # initialize variables for saving important network activities
        RNN_output = torch.zeros((n_time, batch_size, self.hidden_size))

        # initialize RNN and MD activities
        RNN_hidden_t = torch.zeros((1, batch_size, self.hidden_size))
        

        for t in range(n_time):
            input_t = input[t, ...].unsqueeze(dim=0)
            RNN_hidden_t = self.rnn(input_t, RNN_hidden_t)
            RNN_output[t, :, :] = RNN_hidden_t

        model_out = torch.tanh(self.fc(RNN_output))

        return model_out


input_size = 4          # 4 cues
hidden_size = 1000      # number of PFC neurons
output_size = 2         # 2 rules
num_layers = 1
nonlinearity = 'tanh'
Num_MD = 10
num_active = 5
MDeffect = False

# create model
model = Elman_MD(input_size=input_size, hidden_size=hidden_size, output_size=output_size,\
                 num_layers=num_layers, nonlinearity=nonlinearity, Num_MD=Num_MD, num_active=num_active,\
                 tsteps=tsteps, MDeffect=MDeffect)

for name, parm in model.named_parameters():
    print(name)

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.MSELoss()

total_step = sum(blocklen)//batch_size
print_step = 10 # print statistics every print_step
running_loss = 0.0
running_mseloss = 0.0
running_train_time = 0


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
    ###print('MSE', criterion(outputs, labels))
    ###print('reg', disjoint_penalty(model, reg=reg))

    # backward + optimize
    loss = criterion(outputs, labels)
    ###print(loss)
    ###print(model.parm['rnn.input2h.weight'])
    ###print(model.parm['rnn.h2h.weight'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients
    optimizer.step()


    # print statistics
    running_train_time += time.time() - train_time_start
    running_loss += loss.item()
    running_mseloss += loss.item()

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


print('Finished Training')