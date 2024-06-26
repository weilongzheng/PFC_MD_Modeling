''' Defines a gated PFC model that takes an index to retreive the relevant multiplicative gate mask for the task '''

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import stats

class CTRNN_MD(nn.Module):
    """Continuous-time RNN that can take MD inputs.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        sub_size: Number of subpopulation neurons
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, config, dt=100, **kwargs):
        super().__init__()
        self.input_size =  config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.num_task = config.num_task
        self.md_size = config.md_size
        self.use_external_input_mask = config.use_external_inputs_mask
        self.use_multiplicative_gates = config.use_gates
        self.device = config.device
        self.g = .5 # Conductance for recurrents. Consider higher values for realistic richer RNN dynamics, at least initially.

        self.tau = config.tau
        # if dt is None:
            # alpha = 1
        # else:
        alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        if self.use_multiplicative_gates:
            # self.gates = torch.normal(config.md_mean'], 1., size=(config.md_size'], config.hidden_size'], ),
            #   dtype=torch.float) #.type(torch.LongTensor) device=self.device,
            # control density 
            self.gates_mask = np.random.uniform(0, 1, size=(config.md_size, config.hidden_size, )) 
            self.gates_mask = (self.gates_mask < config.MD2PFC_prob).astype(float)
            self.gates_mask = torch.from_numpy(self.gates_mask).to(config.device).float() #* torch.abs(self.gates) 
            self.gates = self.gates_mask
            # self.register_parameter(name='gates', param=torch.nn.Parameter(self.gates_mask))
            # import pdb; pdb.set_trace()
          
            # torch.uniform(config.md_mean'], 1., size=(config.md_size'], config.hidden_size'], ),
            #  device=self.device, dtype=torch.float)
                # *config.G/np.sqrt(config.Nsub*2)
            # Substract mean from each row.
            # self.gates -= np.mean(self.gates, axis=1)[:, np.newaxis]

        # sensory input layer
        self.input2h = nn.Linear(self.input_size, self.hidden_size)

        # hidden layer
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.reset_parameters()


    def reset_parameters(self):
        # identity*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= self.g

    def init_hidden(self, input):
        batch_size = input.shape[1]
        # as zeros
        hidden = torch.zeros(batch_size, self.hidden_size)
        # as uniform noise
        # hidden = 1/self.hidden_size*torch.rand(batch_size, self.hidden_size)
        return hidden.to(input.device)

    def recurrence(self, input, sub_id, hidden):
        """Recurrence helper."""
        ext_input = self.input2h(input)
        rec_input = self.h2h(hidden)

        if self.use_multiplicative_gates:
            #TODO restructure to have sub_id already passed per trial in a batch, and one_hot encoded [batch, md_size]
            batch_sub_encoding = sub_id 
            gates = torch.matmul(batch_sub_encoding.to(self.device), self.gates)
            rec_input = torch.multiply( gates, rec_input)
            # import pdb; pdb.set_trace()
        pre_activation = ext_input + rec_input
        
        h_new = torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)

        return h_new

    def forward(self, input, sub_id, hidden=None):
        """Propogate input through the network."""
        
        num_tsteps = input.size(0)

        # init network activities
        if hidden is None:
            hidden = self.init_hidden(input)
        
        # initialize variables for saving network activities
        output = []
        for i in range(num_tsteps):
            hidden = self.recurrence(input[i], sub_id, hidden)
                
            # save PFC activities
            output.append(hidden)
        
        output = torch.stack(output, dim=0)
        return output, hidden

class RNN_MD(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        sub_size: int, subpopulation size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, config):
        super().__init__()

        self.rnn = CTRNN_MD(config)
        self.drop_layer = nn.Dropout(p=0.05)
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x, sub_id):
        rnn_activity, _ = self.rnn(x, sub_id)
        rnn_activity = self.drop_layer(rnn_activity)
        out = self.fc(rnn_activity)
        return out, rnn_activity
