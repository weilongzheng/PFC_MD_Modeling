import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from pathlib import Path
import json
import time
import math
import itertools
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

'''
source activate pytorch
cd tmp
nohup python -u default_test_twotasks.py > default_test_twotasks.log 2>&1 &
'''

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

# set config
config = {
    # global
     'RNGSEED': 5,
    # envs
     'tasks': ngym.get_collection('yang19'),
     'env_kwargs': {'dt': 100},
     'seq_len': 50,
    # model
     'input_size': 33,
     'hidden_size': 256,
     'sub_size': 128,
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

# set device
device = 'cpu' # always CPU

# set random seed
RNGSEED = config['RNGSEED']
random.seed(RNGSEED)
np.random.seed(RNGSEED)
torch.manual_seed(RNGSEED)

# main loop
count = -1
for tasks in itertools.permutations(config['tasks'], 2):
    count += 1

    # envs for training
    print(tasks)
    envs = []
    for task in tasks:
        env = gym.make(task, **config['env_kwargs'])
        envs.append(env)
    # envs for test
    test_envs = envs

    for MD_flag in ['noMD', 'withMD']:
        if MD_flag == 'noMD':
            config['MDeffect'] = False
        elif MD_flag == 'withMD':
            config['MDeffect'] = True
        
        # model
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
        total_training_cycle = 20
        print_every_cycle = 400
        save_every_cycle = 2000
        running_loss = 0.0
        running_train_time = 0
        log = {
            'tasks': tasks,
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
        
        # save log
        np.save('./files/'+f'{count}_log_' + MD_flag + '.npy', log)


        # Cross Entropy loss
        font = {'family':'Times New Roman','weight':'normal', 'size':20}
        plt.figure()
        plt.plot(np.array(log['losses']))
        plt.xlabel('Trials', fontdict=font)
        plt.ylabel('Training CE loss', fontdict=font)
        plt.tight_layout()
        plt.savefig('./files/'+f'{count}_CEloss_' + MD_flag + '.png')
        plt.close()

    # Task performance with MD & no MD
    log_noMD = np.load('./files/'+f'{count}_log_noMD.npy', allow_pickle=True).item()
    log_withMD = np.load('./files/'+f'{count}_log_withMD.npy', allow_pickle=True).item()
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
    for env_id in range(len(tasks)):
        plt.figure()
        plt.plot(log_noMD['stamps'], log_noMD['act_perfs'][env_id], color='grey', label='$ MD- $')
        plt.plot(log_withMD['stamps'], log_withMD['act_perfs'][env_id], color='red', label='$ MD+ $')
        plt.fill_between(x=[   0,  6000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.fill_between(x=[6000,  12000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
        plt.fill_between(x=[12000, 18000], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.legend(bbox_to_anchor = (1.25, 0.7), prop=legend_font)
        plt.xlabel('Trials', fontdict=label_font)
        plt.ylabel('Performance', fontdict=label_font)
        plt.title('Task{:d}: '.format(env_id+1)+tasks[env_id], fontdict=title_font)
        plt.xlim([0.0, None])
        plt.ylim([0.0, 1.01])
        plt.yticks([0.1*i for i in range(11)])
        plt.tight_layout()
        plt.savefig('./files/'+f'{count}_performance_task_{env_id}.png')
        plt.close()
    
    if count == 1:
        break
