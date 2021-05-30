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
from model_dev import RNN_MD

import matplotlib.pyplot as plt
import seaborn as sns

###--------------------------Helper functions--------------------------###

def get_full_performance(net, env, task_id, num_task, num_trial=1000, device='cpu'):
    fix_perf = 0.
    act_perf = 0.
    num_no_act_trial = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred, _ = net(inputs, sub_id=task_id)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)

        fix_len = sum(gt == 0)
        act_len = len(gt) - fix_len
        assert all(gt[:fix_len] == 0)
        fix_perf += sum(action_pred[:fix_len, 0] == 0)/fix_len
        if act_len != 0:
            assert all(gt[fix_len:] == gt[-1])
            act_perf += sum(action_pred[fix_len:, 0] == gt[-1])/act_len
        else: # no action in this trial
            num_no_act_trial += 1

    fix_perf /= num_trial
    act_perf /= num_trial - num_no_act_trial
    return fix_perf, act_perf


###--------------------------Training configs--------------------------###

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device, '\n')

config = {
    'RNGSEED': 5,
    'env_kwargs': {'dt': 100},
    'lr': 1e-4, # 1e-4 for CTRNN, 1e-3 for LSTM
    'batch_size': 1,
    'seq_len': 100,
    # 'tasks': ngym.get_collection('yang19')
    'tasks': ['yang19.go-v0', 'yang19.rtgo-v0'] # 'tasks': ['yang19.go-v0', 'yang19.dm1-v0']
}

# set random seed
RNGSEED = config['RNGSEED']
np.random.seed([RNGSEED])
torch.manual_seed(RNGSEED)


###--------------------------Generate dataset--------------------------###

tasks = config['tasks']
print(tasks)

# block training - 2 tasks
datasets = []
for task in tasks:
    schedule = RandomSchedule(1)
    env = ScheduleEnvs([gym.make(task, **config['env_kwargs'])], schedule=schedule, env_input=False)
    datasets.append(ngym.Dataset(env, batch_size=config['batch_size'], seq_len=config['seq_len']))
# get envs for test
test_envs = [datasets[env_id].env for env_id in range(len(datasets))]

# only for tasks in Yang19 collection
ob_size = 33
act_size = 17


###--------------------------Generate model--------------------------###

# Model settings
model_config = {
    'input_size': ob_size,
    'hidden_size': 256,
    'sub_size': 128,
    'output_size': act_size,
    'MDeffect': True,
    'md_size': 10,
    'md_active_size': 5,
    'md_dt': 0.001,
}
config.update(model_config)

# RNN_MD model
net = RNN_MD(input_size     = config['input_size' ],
             hidden_size    = config['hidden_size'],
             sub_size       = config['sub_size'],
             output_size    = config['output_size'],
             dt             = config['env_kwargs']['dt'],
             MDeffect       = config['MDeffect'],
             md_size        = config['md_size'],
             md_active_size = config['md_active_size'],
             md_dt          = config['md_dt'],).to(device)
net = net.to(device)
print(net, '\n')


###--------------------------Train network--------------------------###

criterion = nn.MSELoss()

print('training parameters:')
training_params = list()
for name, param in net.named_parameters():
    if True: # 'rnn.input2h' not in name:
        print(name)
        training_params.append(param)
print()
optimizer = torch.optim.Adam(training_params, lr=config['lr'])


total_training_cycle = 1500
print_every_cycle = 50
save_every_cycle = 10
save_times = total_training_cycle//save_every_cycle
running_loss = 0.0
running_train_time = 0
log = {
    'losses': [],
    'stamps': [],
    'fix_perfs': [[], []],
    'act_perfs': [[], []],
    'PFCouts_all': np.zeros(shape=(save_times, config['seq_len'], config['batch_size'], config['hidden_size'])),
}
if config['MDeffect']:
    MD_log = {
        'MDouts_all':               np.zeros(shape=(save_times, config['seq_len'], config['md_size'])),
        'MDpreTraces_all':          np.zeros(shape=(save_times, config['seq_len'], config['hidden_size'])),
        'MDpreTrace_threshold_all': np.zeros(shape=(save_times, config['seq_len'], 1)),
        'wPFC2MD_list': [],
        'wMD2PFC_list': [],
    }
    log.update(MD_log)


for i in range(total_training_cycle):

    train_time_start = time.time()
    
    # control training paradigm
    if i < 500:
        task_id = 0 
    elif i > 500 and i < 1000:
        task_id = 1
    else:
        task_id = 0
    
    dataset = datasets[task_id]
    inputs, labels = dataset()
    assert not np.any(np.isnan(inputs))

    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    # inputs = inputs / (abs(inputs).max() + 1e-15) # normalize inputs
    labels = torch.from_numpy(labels).type(torch.long).to(device) # numpy -> torch
    labels = (F.one_hot(labels, num_classes=act_size)).float() # index -> one-hot vector

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs, rnn_activity = net(inputs, sub_id=task_id)
    # check PFC activities
    # if i % 100 == 99:
    #     plt.plot(rnn_activity[-1, 0, :].detach().numpy())
    #     plt.show()
    # check shapes
    # print("inputs.shape: ", inputs.shape)
    # print("labels.shape: ", labels.shape)
    # print("outputs.shape: ", outputs.shape)

    loss = criterion(outputs, labels)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(training_params, 1.0) # clip the norm of gradients
    optimizer.step()

    # save activities
    if i % save_every_cycle == (save_every_cycle - 1):
        count_save_time = (i+1)//save_every_cycle - 1
        log['PFCouts_all'][count_save_time, ...] = rnn_activity.detach().numpy()
        if config['MDeffect']:
            log['MDouts_all'][count_save_time, ...] = net.rnn.md.md_output_t
            log['MDpreTraces_all'][count_save_time, ...] = net.rnn.md.md_preTraces
            log['MDpreTrace_threshold_all'][count_save_time, ...] = net.rnn.md.md_preTrace_thresholds
            log['wPFC2MD_list'].append(net.rnn.md.wPFC2MD)
            log['wMD2PFC_list'].append(net.rnn.md.wMD2PFC)

    # print statistics
    log['losses'].append(loss.item())
    running_loss += loss.item()
    running_train_time += time.time() - train_time_start
    if i % print_every_cycle == (print_every_cycle - 1):

        print('Total step: {:d}'.format(total_training_cycle))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_every_cycle, i+1))

        # train loss
        print('MSE loss: {:0.9f}'.format(running_loss / print_every_cycle))
        running_loss = 0.0
        
        # test during training
        test_time_start = time.time()
        log['stamps'].append(i+1)
        #   fixation & action performance
        print('Performance')
        for env_id in range(len(datasets)):
            fix_perf, act_perf = get_full_performance(net, test_envs[env_id], task_id=task_id, num_task=len(tasks), num_trial=200, device=device) # set large enough num_trial to get good statistics
            log['fix_perfs'][env_id].append(fix_perf)
            log['act_perfs'][env_id].append(act_perf)
            print('  fix performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, fix_perf))
            print('  act performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, act_perf))
        running_test_time = time.time() - test_time_start

        # left training time
        print('Predicted left training time: {:0.0f} s'.format(
             (running_train_time + running_test_time) * (total_training_cycle - i - 1) / print_every_cycle),
             end='\n\n')
        running_train_time = 0

print('Finished Training')


###--------------------------Analysis--------------------------###

# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':25}
plt.figure()
plt.plot(np.array(log['losses']))
plt.xlabel('Training Cycles', fontdict=font)
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
for env_id in range(len(datasets)):
    plt.figure()
    plt.plot(log['stamps'], log['fix_perfs'][env_id], label='fix')
    plt.plot(log['stamps'], log['act_perfs'][env_id], label='act')
    # plt.axvline(x=5000, c="k", ls="--", lw=1)
    # plt.axvline(x=10000, c="k", ls="--", lw=1)
    plt.legend(prop=legend_font)
    plt.xlabel('Training Cycles', fontdict=label_font)
    plt.ylabel('Performance', fontdict=label_font)
    plt.title('Task{:d}: '.format(env_id+1)+tasks[env_id], fontdict=title_font)
    # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
    plt.ylim([0.0, 1.05])
    plt.yticks([0.1*i for i in range(11)])
    plt.tight_layout()
    # plt.savefig('./animation/'+'performance.png')
    plt.show()

# Heatmap wPFC2MD
# font = {'family':'Times New Roman','weight':'normal', 'size':30}
# ax = plt.figure(figsize=(8, 6))
# ax = sns.heatmap(net.rnn.md.wPFC2MD, cmap='Reds')
# ax.set_xticks([0, 255])
# ax.set_xticklabels([1, 256], rotation=0)
# ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
# ax.set_xlabel('PFC neuron index', fontdict=font)
# ax.set_ylabel('MD neuron index', fontdict=font)
# ax.set_title('wPFC2MD', fontdict=font)
# cbar = ax.collections[0].colorbar
# cbar.set_label('connection weight', fontdict=font)
# plt.tight_layout()
# # plt.savefig('./animation/'+'wPFC2MD.png')
# plt.show()