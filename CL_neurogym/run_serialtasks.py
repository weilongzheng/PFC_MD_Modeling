#!/usr/bin/env python
# coding: utf-8
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
from collections import defaultdict
# computation
import math
import numpy as np
rng = np.random.default_rng()
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
from models.PFC_gated import RNN_MD
from configs.configs import PFCMDConfig, SerialConfig
from logger.logger import SerialLogger
from utils import stats, get_trials_batch, get_performance, accuracy_metric
# visualization
import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from tqdm import tqdm, trange


import argparse
my_parser = argparse.ArgumentParser(description='Train neurogym tasks sequentially')
my_parser.add_argument('exp_name',
                       default='correlated_gates',
                       type=str, nargs='?',
                       help='Experiment name, also used to create the path to save results')
my_parser.add_argument('use_gates',
                       default=1, nargs='?',
                       type=int,
                       help='Use multiplicative gating or not')
my_parser.add_argument('same_rnn',
                       default=1, nargs='?',
                       type=int,
                       help='Train the same RNN for all task or create a separate RNN for each task')
my_parser.add_argument('train_to_criterion',
                       default=1, nargs='?',
                       type=int,
                       help='TODO')
my_parser.add_argument('--experiment_type',
                       default='serial', nargs='?',
                       type=str,
                       help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
my_parser.add_argument('--var1',
                       default=0, nargs='?',
                       type=int,
                       help='Seed')
my_parser.add_argument('--var2',
                       default=-0.3, nargs='?',
                       type=float,
                       help='the ratio of active neurons in gates ')
my_parser.add_argument('--var3',
                        default=100, nargs='?',
                        type=float,
                        help='tau')
my_parser.add_argument('--num_of_tasks',
                    default=30, nargs='?',
                    type=int,
                    help='number of tasks to train on')


# Get args and set config
args = my_parser.parse_args()

exp_name = args.exp_name
os.makedirs('./files/'+exp_name, exist_ok=True)

if args.experiment_type == 'pairs':
    config = PFCMDConfig()
elif args.experiment_type == 'serial':
    config = SerialConfig()

config.set_strings( exp_name)

if args.experiment_type == 'pairs':
    config.task_seq = config.sequences[args.var1]
    config.human_task_names = ['{}'.format(tn[7:-3]) for tn in config.task_seq] #removes yang19 and -v0    {:<6}
    config.exp_signature = f'{config.human_task_names[0]}_{config.human_task_names[1]}_'
elif args.experiment_type == 'serial':
    config.exp_signature = f'{args.var1:d}_{args.var2:1.1f}_{args.var3:1.1f}_'


# config.MDeffect_mul = True if bool(args.var3) else False
# config.MDeffect_add = not config.MDeffect_mul
config.tau= args.var3
config.MD2PFC_prob = args.var2
config.use_gates = bool(args.use_gates)
config.gates_gaussian_cut_off = config.MD2PFC_prob
config.same_rnn = bool(args.same_rnn)
config.train_to_criterion = bool(args.train_to_criterion)

config.exp_signature = config.exp_signature + f'gaus_cut_{"tc" if config.train_to_criterion else "nc"}_{"mul" if config.MDeffect_mul else "add"}_{"gates" if config.use_gates else "nog"}'
config.FILEPATH += exp_name +'/'

config.save_detailed = True
config.use_external_inputs_mask = False
config.MDeffect = False

print(config.task_seq)

###--------------------------Training configs--------------------------###
task_seq = []
# Add tasks gradually with rehearsal 1 2 1 2 3 1 2 3 4 ...
task_sub_seqs = [[(i, config.tasks[i]) for i in range(s)] for s in range(2, len(config.tasks)+1)] # interleave tasks and add one task at a time
for sub_seq in task_sub_seqs: task_seq+=sub_seq
task_seq+=list(reversed(sub_seq)) # One additional final rehearsal, but revearsed for best final score.

# Now adding many random rehearsals:
Random_rehearsals = 0
for _ in range(Random_rehearsals):
    random.shuffle(sub_seq)
    task_seq+=sub_seq

if not args.var1 == 0: # if given seed is not zero, shuffle the task_seq
    #Shuffle tasks
    if True:
        rng = np.random.default_rng(int(args.var1))
        idx = rng.permutation(range(len(config.tasks)))
        config.set_tasks((np.array(config.tasks)[idx]).tolist())
    #Move delayGO and rt GO to args.var1 position:
    # go, rtgo = config.tasks[:2]
    # config.tasks.pop(0)
    # config.tasks.pop(0)
    # config.tasks.insert( args.var1, go)
    # config.tasks.insert( args.var1+1, rtgo)
    

# main loop

def create_model():
    # model
    if config.use_lstm:
        pass
    else:
        net = RNN_MD(config)
    net.to(config.device)
    return(net )


if config.same_rnn:
    net = create_model()

# Replace the gates with the ones calculated offline from performance traces to measure tasks compatibility. 
if config.use_gates:
    gates_tasks = np.load(f'./data/perf_corr_mat_var1_{args.var1}.npy', allow_pickle=True).item()
    gates_corr = gates_tasks['corr_mat'] 
    # assert(config.tasks== gates_tasks['tasks'])  # NOTE using the gates corr off policy here. 
    sampled_gates = np.random.multivariate_normal(np.zeros(net.rnn.gates.shape[0]), gates_corr, net.rnn.gates.shape[1]).T
    sampled_gates = torch.tensor(sampled_gates> config.gates_gaussian_cut_off, device=config.device).float()
    net.rnn.gates = sampled_gates.clone()

# criterion & optimizer
criterion = nn.MSELoss()
#     print('training parameters:')
training_params = list()
for name, param in net.named_parameters():
    print(name)
    training_params.append(param)
optimizer = torch.optim.Adam(training_params, lr=config.lr)


training_log = SerialLogger(config=config)
testing_log = log = SerialLogger(config=config)

envs = []
num_tasks = len(config.tasks)
accuracy_running_average =0.
# Make all tasks
for task_id, task_name in config.tasks_id_name:
    env = gym.make(task_name, **config.env_kwargs)
    envs.append(env)

step_i = 0
bar_tasks = tqdm(task_seq)
for (task_id, task_name) in bar_tasks:
  
    env = envs[task_id]
    bar_tasks.set_description('task: ' + config.human_task_names[task_id])
    testing_log.switch_trialxxbatch.append(step_i)
    testing_log.switch_task_id.append(task_id)
    
    if not config.same_rnn:
        net= create_model()

    # training
    
    if config.MDeffect:
        net.rnn.md.learn = True
        net.rnn.md.sendinputs = True
    running_acc = 0
    training_bar = trange(config.max_trials_per_task//config.batch_size)
    for i in training_bar:
        # control training paradigm
        context_id = task_id if config.use_gates else 0 

        # fetch data
        inputs, labels = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if config.use_lstm:
            outputs, rnn_activity = net(inputs)
        else:
            outputs, rnn_activity = net(inputs, sub_id=context_id)
        # print(f'shape of outputs: {outputs.shape},    and shape of rnn_activity: {rnn_activity.shape}')
        #Shape of outputs: torch.Size([20, 100, 17]),    and shape of rnn_activity: torch.Size ([20, 100, 256
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc  = accuracy_metric(outputs.detach(), labels.detach())

        # save loss

        training_log.write_basic(step_i, loss.item(), acc)
        training_log.gradients.append(np.array([torch.norm(p.grad).item() for p in net.parameters()]) )
        if config.save_detailed:
            training_log.write_detailed( rnn_activity= rnn_activity.detach().cpu().numpy().mean(0),
            inputs=   inputs.detach().cpu().numpy(),
            outputs = outputs.detach().cpu().numpy()[-1, :, :],
            labels =   labels.detach().cpu().numpy()[-1, :, :],
            sampled_act = rnn_activity.detach().cpu().numpy()[:,:, 1:356:36], # Sample about 10 neurons 
            task_id =task_id,
            # rnn_activity.shape             torch.Size([15, 100, 356])
            )

        training_bar.set_description('ls, acc: {:0.3F}, {:0.2F} '.format(loss.item(), acc)+ config.human_task_names[task_id])

        # print statistics
        if step_i % config.print_every_batches == (config.print_every_batches - 1):
            ################################ test during training
            net.eval()
            if config.MDeffect:
                net.rnn.md.learn = False
            with torch.no_grad():
                testing_log.stamps.append(step_i)
                #   fixation & action performance
#                 print('Performance')
                num_tasks = len(config.tasks)
                testing_context_ids = [tin[0] for tin in config.tasks_id_name] if config.use_gates else [context_id]*len(envs)
                fix_perf, act_perf = get_performance(
                    net,
                    envs,
                    context_ids=testing_context_ids,
                    config = config,
                    batch_size = config.test_num_trials,
                    ) 
                
                testing_log.accuracies.append(act_perf)
                testing_log.gradients.append(np.mean(np.stack(training_log.gradients[-config.print_every_batches:]),axis=0))
#                 for env_id in range(num_tasks):
#                     print('  act performance, task #, name {:d} {}, batch# {:d}: {:0.2f}'.format(
#                         env_id+1, config.human_task_names[env_id], i+1,
#                         act_perf[env_id]))
            net.train()
            if config.MDeffect:
                net.rnn.md.learn = True
            #### End testing

        criterion_accuaracy = config.criterion if config.tasks[task_id] not in config.DMFamily else config.criterion_DMfam
        if ((running_acc > criterion_accuaracy) and config.train_to_criterion) or (i+1== config.max_trials_per_task//config.batch_size):
        # switch task if reached the max trials per task, and/or if train_to_criterion then when criterion reached
            # import pdb; pdb.set_trace()
            running_acc = 0.
            if args.experiment_type == 'pairs':
                #move to next task
                task_id = (task_id+1)%2 #just flip to the other task
                print('switching to task: ', task_id, 'at trial: ', i)
            break # stop training current task if sufficient accuracy. Note placed here to allow at least one performance run before this is triggered.

        step_i+=1
        running_acc = 0.7 * running_acc + 0.3 * acc
    #no more than number of blocks specified
    if (args.experiment_type=='pairs') and(len(testing_log.switch_trialxxbatch) > config.num_blocks):
        break

    training_log.sample_input = inputs[0].detach().cpu().numpy().T
    training_log.sample_label = labels[0].detach().cpu().numpy().T
    training_log.sample_output = outputs[0].detach().cpu().numpy().T
testing_log.total_batches = step_i


# In[9]:

import matplotlib
no_of_values = len(config.tasks)
norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
cmap_obj = matplotlib.cm.get_cmap('Set1') # tab20b tab20
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

log = testing_log
log.switch_trialxxbatch.append(log.stamps[-1])
num_tasks = len(config.tasks)
already_seen =[]
title_label = 'Training tasks sequentially ---> \n    ' + config.exp_name
max_x = log.stamps[-1]
fig, axes = plt.subplots(num_tasks+3,1, figsize=[9,7])
for logi in range(num_tasks):
        ax = axes[ logi ] # log i goes to the col direction -->
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([0, max_x])
#         ax.axis('off')
        ax.plot(log.stamps, [test_acc[logi] for test_acc in log.accuracies], linewidth=1)
        ax.plot(log.stamps, np.ones_like(log.stamps)*0.5, ':', color='grey', linewidth=1)
        ax.set_ylabel(config.human_task_names[logi], fontdict={'color': cmap.to_rgba(config.tasks_id_name[logi][0])})
        for ri in range(len(log.switch_trialxxbatch)-1):
                ax.axvspan(log.switch_trialxxbatch[ri], log.switch_trialxxbatch[ri+1], color =cmap.to_rgba(log.switch_task_id[ri]) , alpha=0.2)
for ti, id in enumerate(log.switch_task_id):
    if id not in already_seen:
        already_seen.append(id)
        task_name = config.tasks_id_name[id][1][7:-3]
        axes[0].text(log.switch_trialxxbatch[ti], 1.3, task_name, color= cmap.to_rgba(id) )

gs = np.stack(log.gradients)

print('gradients shape: ', gs.shape) 
glabels = ['inp_w', 'inp_b', 'rnn_w', 'rnn_b', 'out_w', 'out_b']
ax = axes[num_tasks+0]
gi =0
ax.plot(log.stamps, gs[:,gi+1], label= glabels[gi+1])
ax.plot(log.stamps, gs[:,gi], label= glabels[gi])
ax.legend()
ax.set_xlim([0, max_x])
ax = axes[num_tasks+1]
gi =2
ax.plot(log.stamps, gs[:,gi+1], label= glabels[gi+1])
ax.plot(log.stamps, gs[:,gi], label= glabels[gi])
ax.legend()
ax.set_xlim([0, max_x])
ax = axes[num_tasks+2]
gi =4
ax.plot(log.stamps, gs[:,gi+1], label= glabels[gi+1])
ax.plot(log.stamps, gs[:,gi], label= glabels[gi])
ax.legend()
ax.set_xlim([0, max_x])
# ax.set_ylim([0, 0.25])

final_accuracy_average = np.mean(list(testing_log.accuracies[-1].values()))
plt.savefig('./files/'+ config.exp_name+f'/acc_summary_{config.exp_signature}_{step_i}_{final_accuracy_average:1.2f}.jpg', dpi=300)

np.save('./files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy', testing_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/training_log_{config.exp_signature}.npy', training_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/config_{config.exp_signature}.npy', config, allow_pickle=True)
print('testing logs saved to : '+ './files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy')


def show_input_output(input, label, output=None, axes=None):
    if axes is None:
        fig, axes = plt.subplots(3)
                
    no_output = True if output is None else False
    
    axes[0].imshow(input)
    axes[1].imshow(label)
    if output is not None: axes[2].imshow(output)
    
    axes[0].set_xlabel('Time steps')
#     ax.set_ylabel('fr')
#     ax.set_yticks([1, 17, 33, 34, 49])
    
