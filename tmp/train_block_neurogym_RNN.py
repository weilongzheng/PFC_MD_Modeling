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

def get_modelpath(envid):
    # Make local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
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


###--------------------------Training configs--------------------------###

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device, '\n')

env_kwargs = {'dt': 100}
config = {
    'RNGSEED': 5,
    'env_kwargs': env_kwargs,
    'hidden_size': 256,
    'lr': 1e-3,
    'batch_size': 1,
    'seq_len': 200,
}

# set random seed
RNGSEED = config['RNGSEED']
np.random.seed([RNGSEED])
torch.manual_seed(RNGSEED)


###--------------------------Generate dataset--------------------------###

tasks = ['yang19.go-v0', 'yang19.dm1-v0']

datasets = []
for task in tasks:
    schedule = RandomSchedule(1)
    env = ScheduleEnvs([gym.make(task, **config['env_kwargs'])], schedule=schedule, env_input=False)
    datasets.append(ngym.Dataset(env, batch_size=config['batch_size'], seq_len=config['seq_len']))


# for test
envs = [gym.make(task, **config['env_kwargs']) for task in tasks]
schedule = RandomSchedule(len(envs))
test_env = ScheduleEnvs(envs, schedule=schedule, env_input=False)
test_dataset = ngym.Dataset(env, batch_size=config['batch_size'], seq_len=config['seq_len'])
test_env = test_dataset.env

ob_size = 33
act_size = 17


###--------------------------Model configs--------------------------###

# Model settings
model_config = {
    'input_size': ob_size,
    'output_size': act_size
}
config.update(model_config)


###--------------------------Train network--------------------------###

"""Supervised training networks.
Save network in a path determined by environment ID.
Args:
    envid: str, environment ID.
"""

# CTRNN
net = RNNNet(input_size  = config['input_size' ],
             hidden_size = config['hidden_size'],
             output_size = config['output_size'],
             dt=env.dt).to(device)
net = net.to(device)
print(net, '\n')

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

print('training parameters:')
training_params = list()
for name, param in net.named_parameters():
    if True: # 'rnn' not in name:
        print(name)
        training_params.append(param)
print()
optimizer = torch.optim.Adam(training_params, lr=config['lr'])


total_training_cycle = 6000
print_training_cycle = 100
running_loss = 0.0
running_train_time = 0
log = {
    'losses': [],
    'stamp': [],
    'perf': [],
}


for i in range(total_training_cycle):

    train_time_start = time.time()

    if i < 2000:
        task_id = 0
    elif i > 2000 and i < 4000:
        task_id = 1
    else:
        task_id = 0

    inputs, labels = datasets[task_id]()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels).type(torch.long).to(device)
    labels = (F.one_hot(labels, num_classes=act_size)).float()

    # inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    # labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs, _ = net(inputs)
    # check shapes
    # print("inputs.shape: ", inputs.shape)
    # print("labels.shape: ", labels.shape)
    # print("outputs.shape: ", outputs.shape)

    # loss = criterion(outputs.view(-1, act_size), labels)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    log['losses'].append(loss.item())
    running_loss += loss.item()
    running_train_time += time.time() - train_time_start
    if i % print_training_cycle == (print_training_cycle - 1):

        print('Total step: {:d}'.format(total_training_cycle))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_training_cycle, i+1))

        # loss
        print('Cross entropy loss: {:0.8f}'.format(running_loss / print_training_cycle))
        running_loss = 0.0
        
        # task performance
        test_time_start = time.time()
        perf = get_performance(net, test_env, num_trial=200, device=device)
        running_test_time = time.time() - test_time_start
        log['stamp'].append(i+1)
        log['perf'].append(perf)
        print('task performance at {:d} cycle: {:0.2f}'.format(i+1, perf))

        # training time
        print('Predicted left training time: {:0.0f} s'.format(
             (running_train_time + running_test_time) * (total_training_cycle - i - 1) / print_training_cycle),
             end='\n\n')
        running_train_time = 0

print('Finished Training')


# modelpath = get_modelpath(envid)

# save config
# with open(modelpath / 'config.json', 'w') as f:
#     json.dump(config, f)

# save model
# torch.save(net.state_dict(), modelpath / 'net.pth')


###--------------------------Analysis--------------------------###

# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plt.plot(np.array(log['losses'])*1e3)
plt.xlabel('Training Cycles', fontdict=font)
plt.ylabel('MSE loss * $10^{-3}$', fontdict=font)
# plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
# plt.ylim([0.0, 1.0])
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
# plt.savefig('./animation/'+'CEloss.png')
plt.show()

# Task performance during training
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plt.plot(log['stamp'], log['perf'])
plt.xlabel('Training Cycles', fontdict=font)
plt.ylabel('Performance', fontdict=font)
# plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
plt.ylim([0.0, 1.0])
plt.yticks([0.1*i for i in range(11)])
plt.tight_layout()
# plt.savefig('./animation/'+'performance.png')
plt.show()

# def get_performance(net, env, num_trial=1000, device='cpu'):
#     perf = 0
#     for i in range(num_trial):
#         env.new_trial()
#         ob, gt = env.ob, env.gt
#         ob = ob[:, np.newaxis, :]  # Add batch axis
#         inputs = torch.from_numpy(ob).type(torch.float).to(device)

#         action_pred, _ = net(inputs)
#         action_pred = action_pred.detach().cpu().numpy()
#         action_pred = np.argmax(action_pred, axis=-1)
#         perf += gt[-1] == action_pred[-1, 0]

#     perf /= num_trial
#     return perf
