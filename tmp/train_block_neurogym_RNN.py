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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from model_dev import Net, RNNNet


###--------------------------Helper functions--------------------------###

def get_modelpath(env_id):
    # Make local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / env_id
    os.makedirs(path, exist_ok=True)
    return path

def get_performance(net, env, num_trial=1000, device='cpu'):
    perf = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred, _ = net(inputs)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)

        perf += gt[-1] == action_pred[-1, 0]

    perf /= num_trial
    return perf

def get_full_performance(net, env, task_id, num_task, num_trial=1000, device='cpu'):
    fix_perf = 0.
    act_perf = 0.
    num_no_act_trial = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        ob = add_env_input(ob, task_id, num_task)
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred, _ = net(inputs)
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

def get_test_loss(net, env, criterion, num_trial=1000, device='cpu'):
    test_loss = 0.0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt

        ob = ob[:, np.newaxis, :]  # Add batch axis
        ob = torch.from_numpy(ob).type(torch.float).to(device)

        gt = gt[:, np.newaxis]  # Add batch axis
        gt = torch.from_numpy(gt).type(torch.long).to(device) # numpy -> torch
        gt = (F.one_hot(gt, num_classes=act_size)).float() # index -> one-hot vector

        action_pred, _ = net(ob)
        test_loss += criterion(action_pred, gt).item()

    test_loss /= num_trial
    return test_loss

def add_env_input(inputs, task_id, num_task):
    '''
    add rule inputs in block training setting
    '''
    env_inputs = np.zeros((inputs.shape[0], inputs.shape[1], num_task), dtype=inputs.dtype)
    env_inputs[:, :, task_id] = 1.
    inputs = np.concatenate((inputs, env_inputs), axis=-1)
    return inputs


###--------------------------Training configs--------------------------###

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device, '\n')

config = {
    'RNGSEED': 5,
    'env_kwargs': {'dt': 100},
    'hidden_size': 256,
    'lr': 1e-4, # 1e-4 for CTRNN, 1e-3 for LSTM
    'batch_size': 1,
    'seq_len': 100,
    # 'tasks': ngym.get_collection('yang19')
    'tasks': ['yang19.go-v0', 'yang19.dm1-v0']
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

# interleaved training - 20 tasks
# envs = [gym.make(task, **config['env_kwargs']) for task in tasks]
# schedule = RandomSchedule(len(envs))
# env = ScheduleEnvs(envs, schedule=schedule, env_input=True) # env_input should be true
# dataset = ngym.Dataset(env, batch_size=config['batch_size'], seq_len=config['seq_len'])
# env = dataset.env
# test_env = env

# only for tasks in Yang19 collection
ob_size = 33 + len(tasks)
act_size = 17


###--------------------------Generate model--------------------------###

# Model settings
model_config = {
    'input_size': ob_size,
    'output_size': act_size
}
config.update(model_config)

# Elman or LSTM
# net = Net(input_size  = config['input_size' ],
#           hidden_size = config['hidden_size'],
#           output_size = config['output_size'])
# CTRNN model
net = RNNNet(input_size  = config['input_size' ],
             hidden_size = config['hidden_size'],
             output_size = config['output_size'],
             dt=env.dt).to(device)
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


total_training_cycle = 40000
print_training_cycle = 50
running_loss = 0.0
running_train_time = 0
log = {
    'losses': [],
    'stamps': [],
    'perfs': [],
    'fix_perfs': [[], []],
    'act_perfs': [[], []],
    'test_losses': []
}


for i in range(total_training_cycle):

    train_time_start = time.time()

    # control training paradigm
    if i < 6000:
        task_id = 0 
    elif i > 6000 and i < 12000:
        task_id = 1
    else:
        task_id = 0
    
    dataset = datasets[task_id]
    inputs, labels = dataset()
    inputs = add_env_input(inputs, task_id, len(tasks))
    assert not np.any(np.isnan(inputs))

    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    # inputs = inputs / (abs(inputs).max() + 1e-15) # normalize inputs
    labels = torch.from_numpy(labels).type(torch.long).to(device) # numpy -> torch
    labels = (F.one_hot(labels, num_classes=act_size)).float() # index -> one-hot vector

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs, _ = net(inputs)
    
    # check shapes
    # print("inputs.shape: ", inputs.shape)
    # print("labels.shape: ", labels.shape)
    # print("outputs.shape: ", outputs.shape)

    loss = criterion(outputs, labels)
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(training_params, 1.0) # clip the norm of gradients
    optimizer.step()

    # print statistics
    log['losses'].append(loss.item())
    running_loss += loss.item()
    running_train_time += time.time() - train_time_start
    if i % print_training_cycle == (print_training_cycle - 1):

        print('Total step: {:d}'.format(total_training_cycle))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_training_cycle, i+1))

        # train loss
        print('MSE loss: {:0.9f}'.format(running_loss / print_training_cycle))
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
        #   task performance
        # perf = get_performance(net, test_env, num_trial=50, device=device)
        # log['perfs'].append(perf)
        # print('task performance at {:d} cycle: {:0.2f}'.format(i+1, perf))
        #   test loss
        # test_loss = get_test_loss(net, test_env, criterion=criterion, num_trial=50, device=device)
        # log['test_losses'].append(test_loss)
        # print('test MSE loss at {:d} cycle: {:0.9f}'.format(i+1, test_loss))
        running_test_time = time.time() - test_time_start

        # left training time
        print('Predicted left training time: {:0.0f} s'.format(
             (running_train_time + running_test_time) * (total_training_cycle - i - 1) / print_training_cycle),
             end='\n\n')
        running_train_time = 0

print('Finished Training')


# modelpath = get_modelpath(env_id)

# save config
# with open(modelpath / 'config.json', 'w') as f:
#     json.dump(config, f)
# save model
# torch.save(net.state_dict(), modelpath / 'net.pth')


###--------------------------Analysis--------------------------###

# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':25}
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

# Test loss during training
# font = {'family':'Times New Roman','weight':'normal', 'size':30}
# plt.plot(log['stamps'], log['test_losses'])
# plt.xlabel('Training Cycles', fontdict=font)
# plt.ylabel('Test MSE loss', fontdict=font)
# plt.tight_layout()
# plt.show()



###--------------------------Test helper function--------------------------###
# def test_get_full_performance(net, env, task_id, num_task, num_trial=1000, device='cpu'):
#     fix_perf = 0.
#     act_perf = 0.
#     num_no_act_trial = 0
#     for i in range(num_trial):
#         env.new_trial()
#         ob, gt = env.ob, env.gt
#         ob = ob[:, np.newaxis, :]  # Add batch axis
#         ob = add_env_input(ob, task_id, num_task)
#         inputs = torch.from_numpy(ob).type(torch.float).to(device)

#         action_pred, _ = net(inputs)
#         action_pred = action_pred.detach().cpu().numpy()
#         action_pred = np.argmax(action_pred, axis=-1)
#         print(gt.squeeze())
#         print(action_pred.squeeze())

#         fix_len = sum(gt == 0)
#         act_len = len(gt) - fix_len
#         print(fix_len, act_len)
#         assert all(gt[:fix_len] == 0)
#         fix_perf += sum(action_pred[:fix_len, 0] == 0)/fix_len
#         if act_len != 0:
#             assert all(gt[fix_len:] == gt[-1])
#             act_perf += sum(action_pred[fix_len:, 0] == gt[-1])/act_len
#         else: # no action in this trial
#             num_no_act_trial += 1

#     fix_perf /= num_trial
#     act_perf /= num_trial - num_no_act_trial
#     return fix_perf, act_perf
# fix_perf, act_perf = test_get_full_performance(net, test_envs[env_id], task_id=task_id, num_task=len(tasks), num_trial=10, device=device)
# print(fix_perf, act_perf)

# works as expected; see example output below
# ground truth
# action prediction
# fixation length of the trial; action length of the trial
# [0 0 0 0 0 0 0 8 8]
# [0 0 0 0 0 0 0 7 7]
# 7 2
# [0 0 0 0 0 0 0 5 5]
# [0 0 0 0 0 0 0 6 6]
# 7 2
# [ 0  0  0  0  0  0  0  0 14 14]
# [ 0  0  0  0  0  0  0  0 14 14]
# 8 2
# [0 0 0 0 0 0 0 0 0 5 5]
# [0 0 0 0 0 0 0 0 0 5 5]
# 9 2
# [0 0 0 0 0 0 0 0 0 1 1]
# [0 0 0 0 0 0 0 0 0 1 1]
# 9 2
# [0 0 0 0 0 0 0 0 1 1]
# [0 0 0 0 0 0 0 0 1 1]
# 8 2
# [0 0 0 0 0 0 0 7 7]
# [0 0 0 0 0 0 0 2 7]
# 7 2
# [ 0  0  0  0  0  0  0  0  0 14 14]
# [ 0  0  0  0  0  0  0  0  0 14 14]
# 9 2
# [ 0  0  0  0  0  0  0  0  0 15 15]
# [ 0  0  0  0  0  0  0  0  0 15 15]
# 9 2
# [ 0  0  0  0  0  0  0 10 10]
# [0 0 0 0 0 0 0 9 9]
# 7 2
# fixation performance; action performance
# 1.0 0.65