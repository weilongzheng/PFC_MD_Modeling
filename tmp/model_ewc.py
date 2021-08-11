# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys, shelve

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch import autograd
except ImportError:
    print('Torch not available')

# CTRNN with MD layer
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

    def __init__(self, input_size, hidden_size, sub_size, output_size, num_task, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_size = sub_size
        self.output_size = output_size
        self.num_task = num_task

        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        # sensory input layer
        self.input2h = nn.Linear(input_size, hidden_size)

        # hidden layer
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        # identity*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

    def init_hidden(self, input):
        batch_size = input.shape[1]
        hidden = torch.zeros(batch_size, self.hidden_size)
        return hidden.to(input.device)

    def recurrence(self, input, sub_id, hidden):
        """Recurrence helper."""
        ext_input = self.input2h(input)
        rec_input = self.h2h(hidden)
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

    def __init__(self, input_size, hidden_size, sub_size, output_size, num_task, **kwargs):
        super().__init__()

        self.rnn = CTRNN_MD(input_size, hidden_size, sub_size, output_size, num_task, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, sub_id):
        rnn_activity, _ = self.rnn(x, sub_id)
        out = self.fc(rnn_activity)
        return out, rnn_activity


class ElasticWeightConsolidation:
    def __init__(self, model, crit, optimizer, parameters, named_parameters, lr=0.001, weight=1000000, device='cpu'):
        self.model = model
        self.weight = weight
        self.crit = crit
        self.optimizer = optimizer
        self.parameters = parameters
        self.named_parameters = named_parameters
        self.device = device

    def _update_mean_params(self):
        for param_name, param in self.named_parameters.items():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, current_ds, task_id, num_batch):
        log_liklihoods = []
        for i in range(num_batch):
            # fetch data
            current_ds.new_trial()
            ob, gt = current_ds.ob, current_ds.gt
            inputs = torch.from_numpy(ob).type(torch.float).to(self.device)
            labels = torch.from_numpy(gt).type(torch.long).to(self.device)
            inputs = inputs[:, np.newaxis, :]
            outputs, _ = self.model(inputs, sub_id=task_id)
            # compute log_liklihoods
            outputs = F.log_softmax(outputs, dim=-1) # the last dim
            log_liklihoods.append(outputs[:, :, labels])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.parameters)
        _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters.items()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset, task_id, num_batches):
        self._update_fisher_params(dataset, task_id, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.named_parameters.items():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def forward_backward_update(self, input, target, EWC_reg=True):
        output = self.model(input)
        if EWC_reg:
            loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target)
        else:
            loss = self.crit(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)