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
from utils import stats
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
                       default='add_more_reheaerasal',
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
                       default=0.5, nargs='?',
                       type=float,
                       help='the ratio of active neurons in gates ')
my_parser.add_argument('--var3',
                        default=1, nargs='?',
                        type=int,
                        help='0 for additive MD and 1 for multiplicative')
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

config.task_seq = config.sequences[args.var1]
config.human_task_names = ['{}'.format(tn[7:-3]) for tn in config.task_seq] #removes yang19 and -v0    {:<6}
config.exp_signature = f'{config.human_task_names[0]}_{config.human_task_names[1]}_'

config.MDeffect_mul = True if bool(args.var3) else False
config.MDeffect_add = not config.MDeffect_mul
config.MD2PFC_prob = args.var2
config.train_to_criterion = bool(args.train_to_criterion)
config.exp_signature = config.exp_signature + f'LR-3_rehearse_{config.MD2PFC_prob}_{"tc" if config.train_to_criterion else "nc"}_{"mul" if config.MDeffect_mul else "add"}'
config.FILEPATH += exp_name +'/'

config.save_detailed = False
config.same_rnn = bool(args.same_rnn)
config.use_external_inputs_mask = False
config.use_gates = bool(args.use_gates)
config.MDeffect = False


print(config.task_seq)

###--------------------------Training configs--------------------------###

# config = {
#     # exp:
#     'exp_name': exp_name,
#     'save_all': False, 
#     # envs
#     ,
#     'env_kwargs': {'dt': 100},
#     'seq_len': 50,
# # Training
#     'trials_per_task' : 200000,
#     'batch_size' : 50,
#     'train_to_criterion': bool(args.train_to_criterion),
#     'device': device,
# # model
#     'use_lstm': False,
#     'same_rnn' : bool(args.same_rnn), 
#     'use_gates': bool(args.use_gates), 
#     'md_mean' : False, #not used
#     'md_range': args.var2, #0.1
#     'use_external_inputs_mask': False,
#     'input_size': 33,
#     'hidden_size': 256,
#     'sub_size': 128,
#     'output_size': 17,
#     'num_task': 2,
#     'MDeffect': False,
#     'md_size': 15,
#     'md_active_size': 5,
#     'md_dt': 0.001,
# # optimizer
#     'lr': args.var2,#1e-4, # 1e-4 for CTRNN, 1e-3 for LSTM
# }

# config.exp_signature = config.exp_name'] +f'_{args.var1}_{args.var2}_'+\
#     f'{"same_rnn" if config["same_rnn"] else "separate"}_{"gates" if config["use_gates"] else "nogates"}'+\
#         f'_{"tc" if config["train_to_criterion"] else "nc"}'
# print(config.exp_signature)

task_seq = []
# Add tasks gradually with rehearsal 1 2 1 2 3 1 2 3 4 ...
task_sub_seqs = [[(i, config.tasks[i]) for i in range(s)] for s in range(2, len(config.tasks)+1)] # interleave tasks and add one task at a time
for sub_seq in task_sub_seqs: task_seq+=sub_seq

# Just sequence the tasks serially
if not args.var1 == 0: # if given see is not zero, shuffle the task_seq
    #Shuffle tasks
    if True:
        rng = np.random.default_rng(int(args.var1))
        idx = rng.permutation(range(len(config.tasks)))
        config.tasks = (np.array(config.tasks)[idx]).tolist()
    #Move delayGO and rt GO to args.var1 position:
    # go, rtgo = config.tasks[:2]
    # config.tasks.pop(0)
    # config.tasks.pop(0)
    # config.tasks.insert( args.var1, go)
    # config.tasks.insert( args.var1+1, rtgo)
config.human_task_names = [t[7:-3] for t in config.tasks]
    
# simplified_task_seq = [(i, config.tasks[i]) for i in range(len(config.tasks))]
# task_seq = simplified_task_seq
# print('Task seq to be learned: ', task_seq)

# In[3]:

def get_performance(net, envs, context_ids, batch_size=100):
    if type(envs) is not type([]):
        envs = [envs]

    fixation_accuracies = defaultdict()
    action_accuracies = defaultdict()
    for task_i, (context_id, env) in enumerate(zip(context_ids, envs)):
        # import pdb; pdb.set_trace()
        inputs, labels = get_trials_batch(env, batch_size)
        if config.use_lstm:
            action_pred, _ = net(inputs) # shape [500, 10, 17]
        else:
            action_pred, _ = net(inputs, sub_id=context_id) # shape [500, 10, 17]
        ap = torch.argmax(action_pred, -1) # shape ap [500, 10]

        gt = torch.argmax(labels, -1)

        fix_lens = torch.sum(gt==0, 0)
        act_lens = gt.shape[0] - fix_lens 

        fixation_accuracy = ((gt==0)==(ap==0)).sum() / np.prod(gt.shape)## get fixation performance. overlap between when gt is to fixate and when model is fixating
           ## then divide by number of time steps.
        fixation_accuracies[task_i] = fixation_accuracy.detach().cpu().numpy()
        action_accuracy = (gt[-1,:] == ap[ -1,:]).sum() / gt.shape[1] # take action as the argmax of the last time step
        action_accuracies[task_i] = action_accuracy.detach().cpu().numpy()
#         import pdb; pdb.set_trace()
    return((fixation_accuracies, action_accuracies))

# In[5]:

def accuracy_metric(outputs, labels):
    ap = torch.argmax(outputs, -1) # shape ap [500, 10]
    gt = torch.argmax(labels, -1)
    action_accuracy = (gt[-1, :] == ap[-1,:]).sum() / gt.shape[1] # take action as the argmax of the last time step
#     import pdb; pdb.set_trace()
    return(action_accuracy.detach().cpu().numpy())

# In[6]:

def get_trials_batch(envs, batch_size):
    # check if only one env or several and ensure it is a list either way.
    if type(envs) is not type([]):
        envs = [envs]
        
    # fetch and batch data
    obs, gts = [], []
    for bi in range(batch_size):
        env = envs[np.random.randint(0, len(envs))] # randomly choose one env to sample from, if more than one env is given
        env.new_trial()
        ob, gt = env.ob, env.gt
        assert not np.any(np.isnan(ob))
        obs.append(ob), gts.append(gt)
    # Make trials of equal time length:
    obs_lens = [len(o) for o in obs]
    max_len = np.max(obs_lens)
    for o in range(len(obs)):
        while len(obs[o]) < max_len:
            obs[o]= np.insert(obs[o], 0, obs[o][0], axis=0)
#             import pdb; pdb.set_trace()
    gts_lens = [len(o) for o in gts]
    max_len = np.max(gts_lens)
    for o in range(len(gts)):
        while len(gts[o]) < max_len:
            gts[o]= np.insert(gts[o], 0, gts[o][0], axis=0)


    obs = np.stack(obs) # shape (batch_size, 32, 33)
    
    gts = np.stack(gts) # shape (batch_size, 32)

    # numpy -> torch
    inputs = torch.from_numpy(obs).type(torch.float).to(config.device)
    labels = torch.from_numpy(gts).type(torch.long).to(config.device)

    # index -> one-hot vector
    labels = (F.one_hot(labels, num_classes=config.output_size)).float() 
    return (inputs.permute([1,0,2]), labels.permute([1,0,2])) # using time first [time, batch, input]

# In[16]:

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
bar_tasks = enumerate(tqdm(task_seq))
for task_i, (task_id, task_name) in bar_tasks:
  
    env = envs[task_id]
    tqdm.write('learning task:\t ' + config.human_task_names[task_id])
    testing_log.switch_trialxxbatch.append(step_i)
    testing_log.switch_task_id.append(task_id)
    print(f'saved step_i: {step_i}  doing task {task_id}  latest log.stamp: ')
    
    if not config.same_rnn:
        net= create_model()

    # criterion & optimizer
    criterion = nn.MSELoss()


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
        inputs, labels = get_trials_batch(envs=env, batch_size = config.batch_size)

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
            )

        training_bar.set_description('loss, acc: {:0.4F}, {:0.3F}'.format(loss.item(), acc))
#         training_bar.set_description('loss, acc: {:0.3F}, {0.3F}'.format(loss.item(), acc))

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


# In[9]:


num_tasks = len(config.tasks)
title_label = 'Training tasks sequentially ---> \n    ' + config.exp_name
log = testing_log
max_x = log.stamps[-1] #* config.print_every_batches
fig, axes = plt.subplots(num_tasks,1, figsize=[9,7])
for logi in range(num_tasks):
        ax = axes[ logi ] # log i goes to the col direction -->
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([0, max_x])
#         ax.axis('off')
        log = testing_log
        ax.plot(log.stamps, [test_acc[logi] for test_acc in log.accuracies], linewidth=2)
        ax.plot(log.stamps, np.ones_like(log.stamps)*0.5, ':', color='grey', linewidth=0.5)
#         if li == 0: ax.set_title(config.human_task_names[logi])
#         if logi == 0: ax.set_ylabel(config.human_task_names[li])
#         ax.set_yticklabels([]) 
#         ax.set_xticklabels([])
#         if logi== li:
#             ax.axvspan(*ax.get_xlim(), facecolor='grey', alpha=0.2)
#         if li == num_tasks-1 and logi in [num_tasks//2 - 4, num_tasks//2, num_tasks//2 + 4] :
#             ax.set_xlabel('batch #')
# axes[num_tasks-1, num_tasks//2-2].text(-8., -2.5, title_label, fontsize=12)     
# exp_parameters = f'Exp parameters: {config.exp_name}\nRNN: {"same" if config.same_rnn else "separate"}'+'\n'+\
#       f'mul_gate: {"True" if config.use_gates else "False"}\
#           {config.exp_signature}'
# axes[num_tasks-1, 0].text(-7., -2.2, exp_parameters, fontsize=7)     
# # plt.show()
plt.savefig('./files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.jpg')


# In[33]:
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
    


# In[22]:


# fig, axes = plt.subplots(num_tasks,4, figsize=[6,14])
# for logi in range(num_tasks):
#     ax = axes[logi , 0 ]
#     ax.set_ylim([-0.1,1])
#     ax.axis('off')
#     log = training_logs[logi]
#     ax.plot(log['stamps, log['accuracy'])
#     show_input_output(log['sample_input'], log['sample_label'], log['sample_output'], axes = axes[logi,1:])
# plt.show()


# In[ ]:





# In[ ]:


#### draw input output
# plt.close('all')
# fig, axes = plt.subplots(20,3, figsize=[6,14])
# for i in range(20):
#     show_input_output(inputs[i].detach().cpu().numpy().T, labels[i].detach().cpu().numpy().T, outputs[i].detach().cpu().numpy().T, axes=axes[i,:])

# plt.show()

