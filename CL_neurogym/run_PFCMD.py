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
from tqdm import tqdm, trange

from configs.configs import *
from logger.logger import PFCMDLogger
from data.ngym import NGYM
from models.PFCMD import RNN_MD
from utils import set_seed, get_task_id, forward_backward, get_optimizer, test_in_training, get_args_from_parser
from analysis.visualization import plot_rnn_activity, plot_MD_variables, plot_loss, plot_perf, plot_fullperf

# configs
USE_PARSER = True
if USE_PARSER:
    import argparse
    my_parser = argparse.ArgumentParser(description='Train neurogym tasks sequentially')
    args = get_args_from_parser(my_parser)

    exp_name = args.exp_name
    os.makedirs('./files/'+exp_name, exist_ok=True)
    exp_signature = ''.join([str(a) for a in args])
    if len(sys.argv) > 1:   # if arguments passed to the python file 
        config = SerialConfig()
        config.use_gates= bool(args.use_gates)

else:
    config = PFCMDConfig()
print(config.task_seq)

# datasets
dataset = NGYM(config)

# set random seed
set_seed(seed=config.RNGSEED)

# model
net = RNN_MD(input_size       =  config.input_size,
             hidden_size      =  config.hidden_size,
             hidden_ctx_size  =  config.hidden_ctx_size,
             sub_size         =  config.sub_size,
             sub_active_size  =  config.sub_active_size,
             output_size      =  config.output_size,
             MDeffect         =  config.MDeffect,
             md_size          =  config.md_size,
             md_active_size   =  config.md_active_size,
             md_dt            =  config.md_dt,
             config           =  config)
net = net.to(config.device)
print(net, '\n')

# criterion & optimizer
criterion = nn.MSELoss()
optimizer, training_params, named_training_params = get_optimizer(net=net, config=config)

# logger
log = PFCMDLogger(config=config)

# training
task_id = 0
running_loss = 0.0

for i in trange(config.total_trials):

    # control training paradigm
    task_id = get_task_id(config=config, trial_idx=i, prev_task_id=task_id)

    inputs, labels = dataset(task_id=task_id)

    loss, rnn_activity = forward_backward(net=net, opt=optimizer, crit=criterion, inputs=inputs, labels=labels, task_id=task_id)

    # plots
    if i % config.plot_every_trials == config.plot_every_trials-1:
        plot_rnn_activity(rnn_activity)
        if hasattr(config, 'MDeffect'):
            if config.MDeffect:
                plot_MD_variables(net, config)
    # statistics
    log.losses.append(loss.item())
    running_loss += loss.item()
    if i % config.test_every_trials == (config.test_every_trials - 1):
        print('Total trial: {:d}'.format(config.total_trials))
        print('Training sample index: {:d}-{:d}'.format(i+1-config.test_every_trials, i+1))
        # train loss
        print('MSE loss: {:0.9f}'.format(running_loss / config.test_every_trials))
        running_loss = 0.0
        # test during training
        test_in_training(net=net, dataset=dataset, config=config, log=log, trial_idx=i)


# save variables
np.save('./files/'+exp_name+f'/config_{exp_signature}.npy', config)
np.save('./files/'+exp_name+f'/log_{exp_signature}.npy', log)
# log = np.load('./files/'+'log.npy', allow_pickle=True).item()
# config = np.load('./files/'+'config.npy', allow_pickle=True).item()

# visualization
plot_loss(log)
plot_fullperf(config, log)
plot_perf(config, log)
