# system
import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from pathlib import Path
import json
# tools
import time
import itertools
# computation
import math
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
# tasks
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
# models
# from model_dev import RNN_MD
from model_dev import serial_RNN_MD as RNN_MD
from utils import get_full_performance
# visualization
import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

'''
source activate pytorch
cd tmp
nohup python -u default_test_twotasks.py > default_test_twotasks.log 2>&1 &

# Turn off MD by changing config, file name of log & perf
'''


###--------------------------Training configs--------------------------###

# set device
device = 'cpu' # always CPU

# set config
config = {
    # envs
     'tasks': ['yang19.dms-v0',
               'yang19.dnms-v0',
               'yang19.dmc-v0',
               'yang19.dnmc-v0',
               'yang19.dm1-v0',
               'yang19.dm2-v0',
               'yang19.ctxdm1-v0',
               'yang19.ctxdm2-v0',
               'yang19.multidm-v0',
               'yang19.dlygo-v0',
               'yang19.dlyanti-v0',
               'yang19.go-v0',
               'yang19.anti-v0',
               'yang19.rtgo-v0',
               'yang19.rtanti-v0'],
     'env_kwargs': {'dt': 100},
     'seq_len': 50,
    # model
     'input_size': 33,
     'hidden_size': 128,
     'sub_size': 64,
     'output_size': 17,
     'batch_size': 1,
     'num_task': 2,
     'MDeffect': True,
     'md_size': 10,
     'md_active_size': 5,
     'md_dt': 0.001,
    # optimizer
     'lr': 1e-4, # 1e-4 for CTRNN, 1e-3 for LSTM
}

task_pairs = list(itertools.permutations(config['tasks'], 2))
task_pairs = [val for val in task_pairs for i in range(2)]

# main loop
for task_pair_id in range(len(task_pairs)):
    
    # envs for training and test
    task_pair = task_pairs[task_pair_id]
    envs = []
    for task in task_pair:
        env = gym.make(task, **config['env_kwargs'])
        envs.append(env)
    test_envs = envs
    print(task_pair)

    # model
    net = RNN_MD(input_size     = config['input_size'],
                 hidden_size    = config['hidden_size'],
                 sub_size       = config['sub_size'],
                 output_size    = config['output_size'],
                 num_task       = config['num_task'],
                 dt             = config['env_kwargs']['dt'],
                 MDeffect       = config['MDeffect'],
                 md_size        = config['md_size'],
                 md_active_size = config['md_active_size'],
                 md_dt          = config['md_dt'],)
    net = net.to(device)
    print(net)

    # criterion & optimizer
    criterion = nn.MSELoss()
    print('training parameters:')
    training_params = list()
    for name, param in net.named_parameters():
        print(name)
        training_params.append(param)
    optimizer = torch.optim.Adam(training_params, lr=config['lr'])

    # training
    total_training_cycle = 50000
    print_every_cycle = 200
    running_loss = 0.0
    running_train_time = 0
    log = {
        'task_pairs': task_pairs,
        'task_pair': task_pair,
        'losses': [],
        'stamps': [],
        'fix_perfs': [[], []],
        'act_perfs': [[], []],
    }
    if config['MDeffect']:
        net.rnn1.md.learn = True
        net.rnn1.md.sendinputs = True
        net.rnn2.md.learn = True
        net.rnn2.md.sendinputs = True


    for i in range(total_training_cycle):

        train_time_start = time.time()    

        # control training paradigm
        if i == 0:
            task_id = 0
        elif i == 20000:
            task_id = 1
            if config['MDeffect']:
                net.rnn1.md.update_mask()
                net.rnn2.md.update_mask()
        elif i == 40000:
            task_id = 0
            if config['MDeffect']:
                net.rnn1.md.update_mask()
                net.rnn2.md.update_mask()

        # fetch data
        env = envs[task_id]
        env.new_trial()
        ob, gt = env.ob, env.gt
        assert not np.any(np.isnan(ob))

        # numpy -> torch
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
        labels = torch.from_numpy(gt).type(torch.long).to(device)

        # index -> one-hot vector
        labels = (F.one_hot(labels, num_classes=config['output_size'])).float()

        # add batch axis
        inputs = inputs[:, np.newaxis, :]
        labels = labels[:, np.newaxis, :]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, rnn_activity = net(inputs, sub_id=task_id)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # save loss
        log['losses'].append(loss.item())

        # print statistics
        running_loss += loss.item()
        running_train_time += time.time() - train_time_start
        if i % print_every_cycle == (print_every_cycle - 1):

            print('Total trial: {:d}'.format(total_training_cycle))
            print('Training sample index: {:d}-{:d}'.format(i+1-print_every_cycle, i+1))

            # train loss
            print('MSE loss: {:0.9f}'.format(running_loss / print_every_cycle))
            running_loss = 0.0
            
            # test during training
            test_time_start = time.time()
            net.eval()
            if config['MDeffect']:
                net.rnn1.md.learn = False
                net.rnn2.md.learn = False
            with torch.no_grad():
                log['stamps'].append(i+1)
                #   fixation & action performance
                print('Performance')
                for env_id in range(len(task_pair)):
                    fix_perf, act_perf = get_full_performance(net, test_envs[env_id], task_id=env_id, num_task=len(task_pair), num_trial=100, device=device) # set large enough num_trial to get good statistics
                    log['fix_perfs'][env_id].append(fix_perf)
                    log['act_perfs'][env_id].append(act_perf)
                    print('  fix performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, fix_perf))
                    print('  act performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, act_perf))
            net.train()
            if config['MDeffect']:
                net.rnn1.md.learn = True
                net.rnn2.md.learn = True
            running_test_time = time.time() - test_time_start

            # left training time
            print('Predicted left training time: {:0.0f} s'.format(
                (running_train_time + running_test_time) * (total_training_cycle - i - 1) / print_every_cycle),
                end='\n\n')
            running_train_time = 0
        
    # save log
    np.save('./files/'+f'{task_pair_id}_log_MD_'+ str(config['MDeffect'])+'.npy', log)

    # Task performance
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
    for env_id in range(len(task_pair)):
        plt.figure()
        plt.plot(log['stamps'], log['act_perfs'][env_id])
        plt.fill_between(x=[   0,  20000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.fill_between(x=[20000, 40000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
        plt.fill_between(x=[40000, 50000], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.xlabel('Trials', fontdict=label_font)
        plt.ylabel('Performance', fontdict=label_font)
        plt.title('Task{:d}: '.format(env_id+1)+task_pair[env_id], fontdict=title_font)
        plt.xlim([0.0, None])
        plt.ylim([0.0, 1.01])
        plt.yticks([0.1*i for i in range(11)])
        plt.tight_layout()
        plt.savefig('./files/'+f'{task_pair_id}_performance_MD_'+ str(config['MDeffect'])+'_task_{env_id}.png')
        plt.close()
