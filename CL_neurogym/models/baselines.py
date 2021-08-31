from argparse import Namespace
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd


### Elastic Weight Consolidation
### ref https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
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
    
    # CE loss
    def _update_fisher_params(self, current_ds, task_id, num_batch):
        log_liklihoods = []
        for i in range(num_batch):
            # fetch data
            ob, gt = current_ds.new_trial(task_id=task_id)
            inputs = torch.from_numpy(ob).type(torch.float).to(self.device)
            labels = torch.from_numpy(gt).type(torch.long).to(self.device)
            inputs = inputs[:, np.newaxis, :]
            outputs, _ = self.model(inputs)
            # compute log_liklihoods
            outputs = F.log_softmax(outputs, dim=-1) # the last dim
            log_liklihoods.append(torch.flatten(outputs[:, :, labels]))
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.parameters)
        _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters.items()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    # MSE loss
    # def _update_fisher_params(self, current_ds, task_id, num_batch):
    #     log_liklihoods = []
    #     for i in range(num_batch):
    #         # fetch data
    #         inputs, labels = current_ds(task_id=task_id)
    #         outputs, _ = self.model(inputs)
    #         # compute log_liklihoods
    #         outputs = F.mse_loss(outputs, labels)
    #         log_liklihoods.append(outputs)
    #     log_likelihood = torch.mean(torch.stack(log_liklihoods), dim=0)
    #     grad_log_liklihood = autograd.grad(log_likelihood, self.parameters)
    #     _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters.items()]
    #     for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
    #         self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

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
        self.optimizer.zero_grad()
        output, rnn_activity = self.model(input)
        if EWC_reg:
            loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target)
        else:
            loss = self.crit(output, target)
        loss.backward()
        self.optimizer.step()
        return loss, rnn_activity

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)




# More baselines
    # Continual model
    # SI