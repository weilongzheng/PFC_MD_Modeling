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
from model_dev import RNN_MD
# from model_ideal import RNN_MD
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
            # act_perf += sum(action_pred[fix_len:, 0] == gt[-1])/act_len
            act_perf += (action_pred[-1, 0] == gt[-1])
        else: # no action in this trial
            num_no_act_trial += 1

    fix_perf /= num_trial
    act_perf /= num_trial - num_no_act_trial
    return fix_perf, act_perf


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
    # 'tasks': ngym.get_collection('yang19'),
    'tasks': ['yang19.go-v0', 'yang19.rtgo-v0'],
    # 'tasks': ['yang19.go-v0', 'yang19.dlydm1-v0'],
    # 'tasks': ['yang19.dm1-v0', 'yang19.dms-v0'],
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
    'hidden_size': 256,
    'sub_size': 128,
    'output_size': act_size,
    'num_task': len(tasks),
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
             num_task       = config['num_task'],
             dt             = config['env_kwargs']['dt'],
             MDeffect       = config['MDeffect'],
             md_size        = config['md_size'],
             md_active_size = config['md_active_size'],
             md_dt          = config['md_dt'],).to(device)
net = net.to(device)
print(net, '\n')


###--------------------------Train network--------------------------###

# criterion & optimizer
criterion = nn.MSELoss()
print('training parameters:')
training_params = list()
for name, param in net.named_parameters():
    # if 'rnn.h2h' not in name: # reservoir
    if True: # learnable RNN
        print(name)
        training_params.append(param)
print()
optimizer = torch.optim.Adam(training_params, lr=config['lr'])


total_training_cycle = 18000
print_every_cycle = 400
save_every_cycle = 2000
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
if config['MDeffect']:
    MD_log = {
        'MDouts_all':                      [],
        'MDpreTraces_all':                 [],
        'MDpreTraces_binary_all':          [],
        'MDpreTrace_threshold_all':        [],
        'MDpreTrace_binary_threshold_all': [],
        'wPFC2MD_list': [],
        'wMD2PFC_list': [],
    }
    log.update(MD_log)
    net.rnn.md.learn = True
    net.rnn.md.sendinputs = True


for i in range(total_training_cycle):

    train_time_start = time.time()    

    # control training paradigm
    if i < 6000:
        task_id = 0
    elif i >= 6000 and i < 12000:
        task_id = 1
    elif i >= 12000:
        task_id = 0

    # fetch data
    env = envs[task_id]
    env.new_trial()
    ob, gt = env.ob, env.gt
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
    if i % 2000 == 1999:
        font = {'family':'Times New Roman','weight':'normal', 'size':20}
        # PFC activities
        plt.figure()
        plt.plot(rnn_activity[-1, 0, :].cpu().detach().numpy())
        plt.title('PFC activities', fontdict=font)
        plt.show()
        if config['MDeffect']:
            # Presynaptic traces
            plt.figure(figsize=(12, 9))
            plt.subplot(2, 2, 1)
            plt.plot(net.rnn.md.md_preTraces[-1, :])
            plt.axhline(y=net.rnn.md.md_preTrace_thresholds[-1], color='r', linestyle='-')
            plt.title('Pretrace', fontdict=font)
            # Binary presynaptic traces
            sub_size = config['sub_size']
            plt.subplot(2, 2, 2)
            plt.plot(net.rnn.md.md_preTraces_binary[-1, :])
            plt.axhline(y=net.rnn.md.md_preTrace_binary_thresholds[-1], color='r', linestyle='-')
            plt.title( 'Pretrace_binary\n' +
                      f'L: {sum(net.rnn.md.md_preTraces_binary[-1, :sub_size]) / sub_size}; ' +
                      f'R: {sum(net.rnn.md.md_preTraces_binary[-1, sub_size:]) / sub_size}; ' +
                      f'ALL: {sum(net.rnn.md.md_preTraces_binary[-1, :]) / len(net.rnn.md.md_preTraces_binary[-1, :])}',
                      fontdict=font)
            # MD activities
            plt.subplot(2, 2, 3)
            plt.plot(net.rnn.md.md_output_t[-1, :])
            plt.title('MD activities', fontdict=font)
            # Heatmap wPFC2MD
            ax = plt.subplot(2, 2, 4)
            ax = sns.heatmap(net.rnn.md.wPFC2MD, cmap='Reds')
            ax.set_xticks([0, 255])
            ax.set_xticklabels([1, 256], rotation=0)
            ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
            ax.set_xlabel('PFC neuron index', fontdict=font)
            ax.set_ylabel('MD neuron index', fontdict=font)
            ax.set_title('wPFC2MD', fontdict=font)
            cbar = ax.collections[0].colorbar
            cbar.set_label('connection weight', fontdict=font)
            ## Heatmap wMD2PFC
            # font = {'family':'Times New Roman','weight':'normal', 'size':20}
            # ax = plt.subplot(2, 3, 5)
            # ax = sns.heatmap(net.rnn.md.wMD2PFC, cmap='Blues_r')
            # ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
            # ax.set_yticks([0, 255])
            # ax.set_yticklabels([1, 256], rotation=0)
            # ax.set_xlabel('MD neuron index', fontdict=font)
            # ax.set_ylabel('PFC neuron index', fontdict=font)
            # ax.set_title('wMD2PFC', fontdict=font)
            # cbar = ax.collections[0].colorbar
            # cbar.set_label('connection weight', fontdict=font)
            ## Heatmap wMD2PFCMult
            # font = {'family':'Times New Roman','weight':'normal', 'size':20}
            # ax = plt.subplot(2, 3, 6)
            # ax = sns.heatmap(net.rnn.md.wMD2PFCMult, cmap='Reds')
            # ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
            # ax.set_yticks([0, 255])
            # ax.set_yticklabels([1, 256], rotation=0)
            # ax.set_xlabel('MD neuron index', fontdict=font)
            # ax.set_ylabel('PFC neuron index', fontdict=font)
            # ax.set_title('wMD2PFCMult', fontdict=font)
            # cbar = ax.collections[0].colorbar
            # cbar.set_label('connection weight', fontdict=font)
            plt.tight_layout()
            plt.show()
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
        log['PFCouts_all'].append(rnn_activity.cpu().detach().numpy().copy())
        if config['MDeffect']:
            log['MDouts_all'].append(net.rnn.md.md_output_t.copy())
            log['MDpreTraces_all'].append(net.rnn.md.md_preTraces.copy())
            log['MDpreTraces_binary_all'].append(net.rnn.md.md_preTraces_binary.copy())
            log['MDpreTrace_threshold_all'].append(net.rnn.md.md_preTrace_thresholds.copy())
            log['MDpreTrace_binary_threshold_all'].append(net.rnn.md.MDpreTrace_binary_threshold)
            log['wPFC2MD_list'].append(net.rnn.md.wPFC2MD.copy())
            log['wMD2PFC_list'].append(net.rnn.md.wMD2PFC.copy())

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
        if config['MDeffect']:
            net.rnn.md.learn = False
        log['stamps'].append(i+1)
        #   fixation & action performance
        print('Performance')
        for env_id in range(len(tasks)):
            fix_perf, act_perf = get_full_performance(net, test_envs[env_id], task_id=env_id, num_task=len(tasks), num_trial=100, device=device) # set large enough num_trial to get good statistics
            log['fix_perfs'][env_id].append(fix_perf)
            log['act_perfs'][env_id].append(act_perf)
            print('  fix performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, fix_perf))
            print('  act performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, i+1, act_perf))
        if config['MDeffect']:
            net.rnn.md.learn = True
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
    plt.fill_between(x=[   0,  6000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
    plt.fill_between(x=[6000,  12000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
    plt.fill_between(x=[12000, 18000], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
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

# Task performance with MD & no MD
log_noMD = np.load('./files/'+'log_noMD_trials18000.npy', allow_pickle=True).item()
label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
title_font = {'family':'Times New Roman','weight':'normal', 'size':25}
legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
for env_id in range(len(tasks)):
    plt.figure()
    plt.plot(log_noMD['stamps'], log_noMD['act_perfs'][env_id], color='grey', label='$ MD- $')
    plt.plot(log['stamps'], log['act_perfs'][env_id], color='red', label='$ MD+ $')
    plt.fill_between(x=[   0,  6000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
    plt.fill_between(x=[6000,  12000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
    plt.fill_between(x=[12000, 18000], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
    plt.legend(bbox_to_anchor = (1.25, 0.7), prop=legend_font)
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