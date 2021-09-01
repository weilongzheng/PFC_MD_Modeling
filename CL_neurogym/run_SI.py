import os
import sys
from pathlib import Path
import json
import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym
import matplotlib.pyplot as plt
import seaborn as sns

from configs.configs import PFCSIConfig
from logger.logger import BaseLogger
from data.ngym import NGYMDataset
from models.PFC import RNN_MD
from models.baselines import SI
from utils import set_seed, forward_backward, get_optimizer, test_in_training
from analysis.visualization import plot_rnn_activity, plot_loss, plot_perf, plot_fullperf

# configs
config = PFCSIConfig()

# datasets
dataset = NGYMDataset(config)

# set random seed
set_seed(seed=config.RNGSEED)

# model
net = RNN_MD(input_size=config.input_size,
             hidden_size=config.hidden_size,
             output_size=config.output_size)
net = net.to(config.device)
print(net, '\n')

# criterion & optimizer
criterion = nn.MSELoss()
optimizer, training_params, named_training_params = get_optimizer(net=net, config=config)

# SI initialization
assert config.SI, 'Turn on SI'
if config.SI:
    si = SI(backbone=net,
            loss=criterion,
            args=config,
            transform=None,
            opt=optimizer,
            device=config.device,
            parameters=training_params,
            named_parameters=named_training_params)
    net = si.net

# logger
log = BaseLogger()

# training
running_loss = 0.0
running_train_time = 0

for i in range(config.total_trials):

    train_time_start = time.time()    

    # control training paradigm
    if i == config.switch_points[0]:
        task_id = config.switch_taskid[0]
    elif i == config.switch_points[1]:
        task_id = config.switch_taskid[1]
        if config.SI:
            si.end_task()
    elif i == config.switch_points[2]:
        task_id = config.switch_taskid[2]
        if config.SI:
            si.end_task()

    inputs, labels = dataset(task_id=task_id)

    loss, rnn_activity = si.observe(inputs=inputs, labels=labels, not_aug_inputs=None, task_id=task_id)

    # plots
    if i % config.plot_every_trials == config.plot_every_trials-1:
        plot_rnn_activity(rnn_activity)
    # statistics
    log.losses.append(loss.item())
    running_loss += loss.item()
    running_train_time += time.time() - train_time_start
    if i % config.test_every_trials == (config.test_every_trials - 1):
        print('Total trial: {:d}'.format(config.total_trials))
        print('Training sample index: {:d}-{:d}'.format(i+1-config.test_every_trials, i+1))
        # train loss
        print('MSE loss: {:0.9f}'.format(running_loss / config.test_every_trials))
        running_loss = 0.0
        # test during training
        test_time_start = time.time()
        test_in_training(net=net, dataset=dataset, config=config, log=log, trial_idx=i)
        running_test_time = time.time() - test_time_start
        # left training time
        print('Predicted left training time: {:0.0f} s'.format(
             (running_train_time + running_test_time) * (config.total_trials - i - 1) / config.test_every_trials),
             end='\n\n')
        running_train_time = 0

# save variables
# np.save('./files/'+'config.npy', config)
# np.save('./files/'+'log.npy', log)
# log = np.load('./files/'+'log.npy', allow_pickle=True).item()
# config = np.load('./files/'+'config.npy', allow_pickle=True).item()

# visualization
plot_loss(log)
plot_fullperf(config, log)
plot_perf(config, log)