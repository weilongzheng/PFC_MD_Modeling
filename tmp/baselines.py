import sys, shelve
from argparse import Namespace
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from utils import get_device
import matplotlib.pyplot as plt


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
            current_ds.new_trial()
            ob, gt = current_ds.ob, current_ds.gt
            ob[:, 1:] = (ob[:, 1:] - np.min(ob[:, 1:]))/(np.max(ob[:, 1:]) - np.min(ob[:, 1:]))
            inputs = torch.from_numpy(ob).type(torch.float).to(self.device)
            labels = torch.from_numpy(gt).type(torch.long).to(self.device)
            inputs = inputs[:, np.newaxis, :]
            outputs, _ = self.model(inputs, sub_id=task_id)
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
        # log_liklihoods = []
        # for i in range(num_batch):
        #     # fetch data
        #     current_ds.new_trial()
        #     ob, gt = current_ds.ob, current_ds.gt
        #     ob[:, 1:] = (ob[:, 1:] - np.min(ob[:, 1:]))/(np.max(ob[:, 1:]) - np.min(ob[:, 1:]))
        #     inputs = torch.from_numpy(ob).type(torch.float).to(self.device)
        #     labels = torch.from_numpy(gt).type(torch.long).to(self.device)
        #     labels = (F.one_hot(labels, num_classes=self.model.rnn.output_size)).float()
        #     inputs = inputs[:, np.newaxis, :]
        #     labels = labels[:, np.newaxis, :]
        #     outputs, _ = self.model(inputs, sub_id=task_id)
        #     # compute log_liklihoods
        #     outputs = F.mse_loss(outputs, labels)
        #     log_liklihoods.append(outputs)
        # log_likelihood = torch.mean(torch.stack(log_liklihoods), dim=0)
        # grad_log_liklihood = autograd.grad(log_likelihood, self.parameters)
        # _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters.items()]
        # for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
        #     self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

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


### other baselines
### ref https://github.com/aimagelab/mammoth

# Arguments class
class Args:
    def __init__(self, 
                 lr=None, c=None, xi=None,
                 e_lambda=None, gamma=None,
                 batch_size=None, n_epochs=None) -> None:
        
        self.lr = lr
        self.c = c
        self.xi = xi
        self.e_lambda = e_lambda
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_epochs = n_epochs

# Continual learning model API
class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                       args: Namespace, transform: torchvision.transforms, opt, device) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = opt
        self.device = device
        self.net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.net.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                   + 100 * output_size + output_size)
        """
        grads = []
        for pp in list(self.net.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

# Synaptic Intelligence
class SI(ContinualModel):
    NAME = 'si'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, opt, device):
        super(SI, self).__init__(backbone, loss, args, transform, opt, device)

        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.get_params().data - self.checkpoint) ** 2 + self.args.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs, labels, not_aug_inputs, sub_id):
        self.opt.zero_grad()
        outputs, rnn_activity = self.net(inputs, sub_id=sub_id)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.args.c * penalty
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        self.opt.step()

        self.small_omega += self.args.lr * self.get_grads().data ** 2

        return loss, rnn_activity