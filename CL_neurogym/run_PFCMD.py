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
from analysis.visualization import plot_rnn_activity, plot_MD_variables, plot_loss, plot_perf, plot_fullperf, plot_both_perf, plot_md_activity

# configs
config = PFCMDConfig()

USE_PARSER = True
if USE_PARSER:
    import argparse
    my_parser = argparse.ArgumentParser(description='Train neurogym tasks sequentially')
    args = get_args_from_parser(my_parser)

    exp_name = args.exp_name
    config.task_seq = config.sequences[args.var1]
    config.set_task_seq(config.task_seq)
    config.human_task_names = ['{}'.format(tn[7:-3]) for tn in config.task_seq] #removes yang19 and -v0    {:<6}
    config.EXPSIGNATURE = f'{config.human_task_names[0]}_{config.human_task_names[1]}_'
    
    config.MDeffect_mul = True if bool(args.var3) else False
    config.MDeffect_add = not config.MDeffect_mul
    # config.MD2PFC_prob = args.var2
    config.gates_sparsity = args.var2
    config.gates_std = args.var4

    config.train_to_criterion = bool(args.train_to_criterion)
    config.EXPSIGNATURE += f'Gm_{config.gates_mean}_s_{config.gates_std}_s_{config.gates_sparsity}' 
    config.EXPSIGNATURE = config.EXPSIGNATURE + f'MDprob_{config.MD2PFC_prob}_{"tc" if config.train_to_criterion else "nc"}_{"mul" if config.MDeffect_mul else "add"}'
    config.FILEPATH += exp_name +'/'
    os.makedirs(config.FILEPATH, exist_ok=True)

# config.total_trsacct
# ials = 1000 # when needing to adjust workflow and debug
    # os.makedirs('./files/'+exp_name, exist_ok=True)
    # exp_signature = ''.join([str(a) for a in args])
    # if len(sys.argv) > 1:   # if arguments passed to the python file 
    #     config = SerialConfig()
    #     config.use_gates= bool(args.use_gates)
print(config.task_seq)

# datasets
dataset = NGYM(config)

# set random seed
set_seed(seed=config.RNGSEED)

# model
net = RNN_MD(config           =  config)
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
running_train_time = 0

accuracy_running_average = 0.
config.switch_points = [0]
config.switch_taskid = [task_id]
tbar = trange(config.total_trials)
for i in tbar:

    train_time_start = time.time()    

    # control training paradigm
    # task_id = get_task_id(config=config, trial_idx=i, prev_task_id=task_id)

    criterion_acc = config.criterion if config.task_seq[task_id] not in config.DMFamily else config.criterion_DMfam
    # switch task if reached the max trials per task, and/or if train_to_criterion then when criterion reached
    if (config.train_to_criterion and(accuracy_running_average > criterion_acc)) or (i-config.switch_points[-1]> config.max_trials_per_task):
        accuracy_running_average = 0.
        config.switch_points.append(i)
        #move to next task
        task_id = (task_id+1)%2 #just flip to the other task
        config.switch_taskid.append(task_id)
        print('switching to task: ', task_id, 'at trial: ', i)
        #but only do this for no more than number of blocks specified
        if len(config.switch_points) > config.num_blocks:
            break

    inputs, labels = dataset(task_id=task_id)
    
    if i == 40000:
        net.rnn.plot_pre_dist = True

    loss, rnn_activity = forward_backward(net=net, opt=optimizer, crit=criterion, inputs=inputs, labels=labels, task_id=task_id)
    net.rnn.plot_pre_dist = False
    # plots
    if i % config.plot_every_trials == config.plot_every_trials-1:
        plot_rnn_activity(rnn_activity, config)
        if hasattr(config, 'MDeffect'):
            if config.MDeffect:
                plot_MD_variables(net, config)
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
        current_acc = log.act_perfs[task_id][-1]
        accuracy_running_average = config.accuracy_momentum*accuracy_running_average+\
            (1-config.accuracy_momentum)*current_acc
        tbar.set_description('loss, acc, task: {:0.4F}, {:0.3F}, {}'.format(running_loss / config.test_every_trials, accuracy_running_average, task_id))
        print('switch points:', config.switch_points)
        running_test_time = time.time() - test_time_start
        # left training time
        print('Predicted left training time: {:0.0f} s'.format(
             (running_train_time + running_test_time) * (config.total_trials - i - 1) / config.test_every_trials),
             end='\n\n')
        running_train_time = 0

# save variables
np.save(config.FILEPATH+'config_' + config.EXPSIGNATURE + '.npy', config)
np.save(config.FILEPATH+'log_' + config.EXPSIGNATURE + '.npy', log)
# log = np.load('./files/'+'log.npy', allow_pickle=True).item()
# config = np.load('./files/'+'config.npy', allow_pickle=True).item()

# visualization
# plot_loss(config, log)
# plot_fullperf(config, log)
# plot_perf(config, log)
plot_both_perf(config, log, net)
# plot_md_activity(net, log, config)

# plot the recurrent weights
plt.close('all')
nw = [np for np in net.named_parameters() if 'h2h' in np[0]]
plt.matshow(nw[0][1].detach().cpu().numpy())
plt.gca().set_title(nw[0][0])
plt.savefig('end_of_training_weights.jpg')
np.save( config.FILEPATH+'weights_' + config.EXPSIGNATURE + '.npy', nw[0][1].detach().cpu().numpy())
print('weights at the end')
print( nw[0][1].detach().cpu().numpy()[:20,:10])
