import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from pathlib import Path
import json
import time
import math
import numpy as np
import random
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from utils import get_full_performance
from model_dev import RNN_MD
# from model_ideal import RNN_MD
from model_ewc import ElasticWeightConsolidation
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
from pygifsicle import optimize


###--------------------------Training configs--------------------------###

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # always CPU

print("device:", device, '\n')

config = {
    'RNGSEED': 5,
    'env_kwargs': {'dt': 100},
    'lr': 1e-4, # 1e-4 for CTRNN, 1e-3 for LSTM
    'batch_size': 1,
    'seq_len': 50,
    'EWC': True,
    'EWC_weight': 1e6,

    # 'tasks': ngym.get_collection('yang19'),
    # 'tasks': ['yang19.go-v0', 'yang19.rtgo-v0'],
    # 'tasks': ['yang19.dms-v0', 'yang19.dmc-v0'],
    # 'tasks': ['yang19.dnms-v0', 'yang19.dnmc-v0'],
    # 'tasks': ['yang19.dlygo-v0', 'yang19.dnmc-v0'],
    'tasks': ['yang19.dlyanti-v0', 'yang19.dnms-v0'],
    # 'tasks': ['yang19.dlyanti-v0', 'yang19.dms-v0'],
    # 'tasks': ['yang19.dm1-v0', 'yang19.dmc-v0'],
}

# set random seed
RNGSEED = config['RNGSEED']
random.seed(RNGSEED)
np.random.seed(RNGSEED)
torch.manual_seed(RNGSEED)


###--------------------------Generate dataset--------------------------###

tasks = config['tasks']
print(tasks)

# block training - 2 tasks
envs = []
for task in tasks:
    env = gym.make(task, **config['env_kwargs'])
    envs.append(env)
# get envs for test
test_envs = envs

# only for tasks in Yang19 collection
ob_size = 33
act_size = 17


###--------------------------Generate model--------------------------###

# Model settings
model_config = {
    'input_size': ob_size,
    'hidden_size': 400,
    'sub_size': 200,
    'output_size': act_size,
    'num_task': len(tasks),
    'MDeffect': False,
    'md_size': 4,
    'md_active_size': 2,
    'md_dt': 0.001,
}
config.update(model_config)

# RNN_MD model
net = RNN_MD(input_size     = config['input_size' ],
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
print(net, '\n')

# init_inweight = net.rnn.input2h.weight.clone()
# init_recweight = net.rnn.h2h.weight.clone()
# init_outweight = net.fc.weight.clone()

# criterion & optimizer
criterion = nn.MSELoss()
print('training parameters:')
training_params = list()
named_training_params = dict()
for name, param in net.named_parameters():
    # if True: # learnable RNN
    if 'rnn.input2PFCctx' not in name:
        print(name)
        training_params.append(param)
        named_training_params[name] = param
optimizer = torch.optim.Adam(training_params, lr=config['lr'])

# EWC
if config['EWC']:
    ewc = ElasticWeightConsolidation(net,
                                     crit=criterion,
                                     optimizer=optimizer,
                                     parameters=training_params,
                                     named_parameters=named_training_params,
                                     lr=config['lr'],
                                     weight=config['EWC_weight'],
                                     device=device)

###--------------------------Train network--------------------------###

total_training_cycle = 50000
print_every_cycle = 500
save_every_cycle = 10000
save_times = total_training_cycle//save_every_cycle
running_loss = 0.0
running_train_time = 0
log = {
    'losses': [],
    'stamps': [],
    'fix_perfs': [[], []],
    'act_perfs': [[], []],
    'PFCouts_all': [],
}


for i in range(total_training_cycle):

    train_time_start = time.time()    

    # control training paradigm
    if i == 0:
        task_id = 0
    elif i == 20000:
        task_id = 1
        if config['EWC']:
            ewc.register_ewc_params(dataset=envs[0], task_id=task_id, num_batches=3000)
    elif i == 40000:
        task_id = 0
        if config['EWC']:
            ewc.register_ewc_params(dataset=envs[1], task_id=task_id, num_batches=3000)

    # fetch data
    env = envs[task_id]
    env.new_trial()
    ob, gt = env.ob, env.gt
    ob[:, 1:] = (ob[:, 1:] - np.min(ob[:, 1:]))/(np.max(ob[:, 1:]) - np.min(ob[:, 1:]))
    assert not np.any(np.isnan(ob))

    # numpy -> torch
    inputs = torch.from_numpy(ob).type(torch.float).to(device)
    labels = torch.from_numpy(gt).type(torch.long).to(device)

    # index -> one-hot vector
    labels = (F.one_hot(labels, num_classes=act_size)).float()

    # add batch axis
    inputs = inputs[:, np.newaxis, :]
    labels = labels[:, np.newaxis, :]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs, rnn_activity = net(inputs, sub_id=task_id)

    # plot during training
    if i % 4000 == 3999:
        font = {'family':'Times New Roman','weight':'normal', 'size':20}
        # PFC activities
        plt.figure()
        plt.plot(rnn_activity[-1, 0, :].cpu().detach().numpy())
        plt.title('PFC activities', fontdict=font)
        plt.show()

    if config['EWC']:
        loss = criterion(outputs, labels) + ewc._compute_consolidation_loss(weight=config['EWC_weight'])
    else:
        loss = criterion(outputs, labels)
    
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(training_params, 1.0) # clip the norm of gradients
    optimizer.step()

    # save activities
    if i % save_every_cycle == (save_every_cycle - 1):
        log['PFCouts_all'].append(rnn_activity.cpu().detach().numpy().copy())

    # print statistics
    log['losses'].append(loss.item())
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
        with torch.no_grad():
            log['stamps'].append(i+1)
            #   fixation & action performance
            print('Performance')
            for env_id in range(len(tasks)):
                fix_perf, act_perf = get_full_performance(net, test_envs[env_id], task_id=env_id, num_task=len(tasks), num_trial=100, device=device) # set large enough num_trial to get good statistics
                log['fix_perfs'][env_id].append(fix_perf)
                log['act_perfs'][env_id].append(act_perf)
                print('  fix performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, fix_perf))
                print('  act performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, act_perf))
        net.train()
        running_test_time = time.time() - test_time_start

        # left training time
        print('Predicted left training time: {:0.0f} s'.format(
             (running_train_time + running_test_time) * (total_training_cycle - i - 1) / print_every_cycle),
             end='\n\n')
        running_train_time = 0

print('Finished Training')


###--------------------------Analysis--------------------------###

# save log
# np.save('./files/'+'log_withMD.npy', log)

# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':25}
plt.figure()
plt.plot(np.array(log['losses']))
plt.xlabel('Trials', fontdict=font)
plt.ylabel('Training MSE loss', fontdict=font)
# plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
# plt.ylim([0.0, 1.0])
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
# plt.savefig('./animation/'+'CEloss.png')
plt.show()

# Task performance during training
label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
title_font = {'family':'Times New Roman','weight':'normal', 'size':25}
legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
for env_id in range(len(tasks)):
    plt.figure()
    plt.plot(log['stamps'], log['fix_perfs'][env_id], label='fix')
    plt.plot(log['stamps'], log['act_perfs'][env_id], label='act')
    plt.fill_between(x=[   0,  15000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
    plt.fill_between(x=[15000, 30000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
    plt.fill_between(x=[30000, 40000], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
    plt.legend(bbox_to_anchor = (1.15, 0.7), prop=legend_font)
    plt.xlabel('Trials', fontdict=label_font)
    plt.ylabel('Performance', fontdict=label_font)
    plt.title('Task{:d}: '.format(env_id+1)+tasks[env_id], fontdict=title_font)
    # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
    plt.xlim([0.0, None])
    plt.ylim([0.0, 1.01])
    plt.yticks([0.1*i for i in range(11)])
    plt.tight_layout()
    # plt.savefig('./animation/'+'performance.png')
    plt.show()

# Task performance EWC & no MD
log_noMD = np.load('./files/'+'log_withMD_trials18000.npy', allow_pickle=True).item()
label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
title_font = {'family':'Times New Roman','weight':'normal', 'size':25}
legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
for env_id in range(len(tasks)):
    plt.figure()
    plt.plot(log_noMD['stamps'], log_noMD['act_perfs'][env_id], color='grey', label='$ MD- $')
    plt.plot(log['stamps'], log['act_perfs'][env_id], color='red', label='$ EWC $')
    plt.fill_between(x=[   0,  20000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
    plt.fill_between(x=[20000, 40000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
    plt.fill_between(x=[40000, 50000], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
    plt.legend(bbox_to_anchor = (1.3, 0.7), prop=legend_font)
    plt.xlabel('Trials', fontdict=label_font)
    plt.ylabel('Performance', fontdict=label_font)
    plt.title('Task{:d}: '.format(env_id+1)+tasks[env_id], fontdict=title_font)
    # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
    plt.xlim([0.0, None])
    plt.ylim([0.0, 1.01])
    plt.yticks([0.1*i for i in range(11)])
    plt.tight_layout()
    # plt.savefig('./animation/'+'performance.png')
    plt.show()