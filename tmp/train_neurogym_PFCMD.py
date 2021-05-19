import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
sys.path.append('D:\\DESKTOP\\Lab\\Projects\\Yang Lab\\neurogym') # directory of local neurogym module
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import gym
import neurogym as ngym
from model_dev import PytorchPFCMD

import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from pygifsicle import optimize



# def get_modelpath(envid):
#     # Make local file directories
#     path = Path('.') / 'files'
#     os.makedirs(path, exist_ok=True)
#     path = path / envid
#     os.makedirs(path, exist_ok=True)
#     return path

# modelpath = get_modelpath(envid)

# print("input size: ", env.observation_space.shape)
# print("output size: ", env.action_space.n)
# print("inputs.shape: ", inputs.shape)
# print("labels.shape: ", labels.shape)
# print("outputs.shape: ", outputs.shape)

# 'PerceptualDecisionMaking-v0' input 3 output 3
# 'DelayMatchCategory-v0' input 3 output 3
# 'DelayMatchSample-v0' input 3 output 3
# 'MultiSensoryIntegration-v0' input 5 output 3
# 'DelayComparison-v0' input 2 output 3
# 'ContextDecisionMaking-v0' input 7 output 3
# 'DualDelayMatchSample-v0' input 7 output 3


###--------------------------Training configs--------------------------###

device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = {
    'RNGSEED': 5,
    'hidden_size': 64,
    'lr': 1e-2,
    'batch_size': 1,
    'seq_len': 200,
    'tasks': ['yang19.dm1-v0', 'yang19.ctxdm1-v0'],
}

tasks = config['tasks']
print('Training paradigm: task1 -> task2 -> task1')
print('Training task1', tasks[0])
print('Training task2', tasks[1], '\n')

env_kwargs = {'dt': 100}
config['env_kwargs'] = env_kwargs

# set random seed
RNGSEED = config['RNGSEED']
np.random.seed([RNGSEED])
torch.manual_seed(RNGSEED)


###--------------------------Generate dataset--------------------------###

envs = [gym.make(task, **config['env_kwargs']) for task in tasks]

# Supervised dataset list
datasets = []
for i in range(len(envs)):
    dataset = ngym.Dataset(envs[i], env_kwargs=env_kwargs, \
                                    batch_size=config['batch_size'], \
                                    seq_len=config['seq_len'])
    datasets.append(dataset)

# observation space
# ob_size_list = [ datasets[i].env.observation_space.shape[0] for i in range(len(datasets)) ]
# ob_size = sum(ob_size_list)
ob_size_per_task = 33
ob_size = 33 * len(datasets)

# action space
# act_size = [ datasets[i].env.action_space.n for i in range(len(datasets)) ]
# assert len(np.unique(act_size)) == 1 # the action spaces should be the same
# act_size = np.unique(act_size)[0]
act_size = 17

# Get data of a trial
def get_data(datasets, ob_size_per_task, ob_size, act_size, seq_len, envid):
    Nevns = len(datasets)
    inputs = np.zeros(shape=(seq_len, config['batch_size'], ob_size))
    inputs[:, :, ob_size_per_task*envid:ob_size_per_task*(envid+1)], labels = datasets[envid]()

    return inputs, labels

# check shapes
# inputs, labels = get_data(datasets=datasets, ob_size_per_task=ob_size_per_task,\
#                           ob_size=ob_size, act_size=act_size,\
#                           seq_len=config['seq_len'], envid=0)
# print(inputs.shape, labels.shape)

###--------------------------Model configs--------------------------###

# Model settings
model_config = {
    'Ntasks': len(tasks),
    'input_size_per_task': ob_size_per_task,
    'n_neuron': 1000,
    'n_neuron_per_cue': 400,
    'Num_MD': 10,
    'num_active': 5, # num MD active per context
    'n_output': act_size,
    'MDeffect': True,
    'PFClearn': False,
}
config.update(model_config)

PFClearn = config['PFClearn']

###--------------------------Train network--------------------------###

model = PytorchPFCMD(Ntasks=config['Ntasks'], input_size_per_task=config['input_size_per_task'], \
                     Num_PFC=config['n_neuron'], n_neuron_per_cue=config['n_neuron_per_cue'], \
                     Num_MD=config['Num_MD'], num_active=config['num_active'], \
                     num_output=config['n_output'], MDeffect=config['MDeffect'])
print(model, '\n')

# check senory input layer
# font = {'family':'Times New Roman','weight':'normal', 'size':20}
# plt.figure(figsize=(15, 10))
# ax = sns.heatmap(model.sensory2pfc.wIn, cmap='Reds')
# ax.set_xlabel('Input unit index', fontdict=font)
# ax.set_ylabel('PFC neuron index', fontdict=font)
# ax.set_title('$ W_{input} $', fontdict=font)
# ax.set_xticks([0, 65])
# ax.set_xticklabels([1, 66], rotation=0)
# ax.set_yticks([0, 399, 799, 999])
# ax.set_yticklabels([1, 400, 800, 1000], rotation=0)
# cbar = ax.collections[0].colorbar
# cbar.set_label('connection weight', fontdict=font)
# plt.tight_layout()
# plt.show()

criterion = nn.CrossEntropyLoss()

print('training parameters:')
training_params = list()
for name, param in model.named_parameters():
    print(name)
    training_params.append(param)
if PFClearn == True:
    print('pfc.Jrec')
    print('\n', end='')
    training_params.append(model.pfc.Jrec)
else:
    print('\n', end='')
optimizer = torch.optim.Adam(training_params, lr=config['lr'])


total_training_cycle = 150
print_training_cycle = 10
running_loss = 0.0
running_train_time = 0
log = {
    'losses': []
}


for i in range(total_training_cycle):

    train_time_start = time.time()

    if i < 50:
        envid = 0
    elif i >= 50 and i < 100:
        envid = 1
    elif i >= 100:
        envid = 0

    inputs, labels = get_data(datasets=datasets, ob_size_per_task=ob_size_per_task,\
                              ob_size=ob_size, act_size=act_size,\
                              seq_len=config['seq_len'], envid=envid)
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)[:, 0, :] # batch_size should be 1
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    outputs = model(inputs, labels)

    # print(inputs.shape, outputs.shape, labels.shape)

    # save PFC and MD activities
    # PFCouts_all[i,:] = model.pfc.activity.detach().numpy()
    # if  MDeffect == True:
    #     MDouts_all[i,:] = model.md_output
    #     MDpreTraces[i,:] = model.md.MDpreTrace
    # for itrial in range(inpsPerConext): 
        #PFCouts_all[i*inpsPerConext+tstart,:,:] = model.pfc_outputs.detach().numpy()[tstart*tsteps:(tstart+1)*tsteps,:]
    # save PFC and MD activities
    # PFCouts_all[i,:,:] = model.pfc_outputs.detach().numpy()
    # if  MDeffect == True:
    #     # MDouts_all[i*inpsPerConext+tstart,:,:] = model.md_output_t[tstart*tsteps:(tstart+1)*tsteps,:]
    #     MDouts_all[i,:,:] = model.md_output_t
    #     MDpreTraces_all[i,:,:] = model.md_preTraces
    #     MDpreTrace_threshold_all[i, :, :] = model.md_preTrace_thresholds

    # backward + optimize
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the norm of gradients 
    if PFClearn == True:
        torch.nn.utils.clip_grad_norm_(model.pfc.Jrec, 1e-6) # clip the norm of gradients; Jrec 1e-6
    optimizer.step()


    # print statistics
    log['losses'].append(loss.item())
    running_loss += loss.item()
    running_train_time += time.time() - train_time_start
    if i % print_training_cycle == (print_training_cycle - 1):

        print('Total step: {:d}'.format(total_training_cycle))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_training_cycle, i+1))

        # loss
        print('Cross entropy loss: {:0.5f}'.format(running_loss / print_training_cycle))
        running_loss = 0.0

        # training time
        print('Predicted left training time: {:0.0f} s'.format(
            (running_train_time) * (total_training_cycle - i - 1) / print_training_cycle),
            end='\n\n')
        running_train_time = 0


print('Finished Training')


# save model
# torch.save(model.state_dict(), modelpath / 'model.pth')

# save config
# with open(modelpath / 'config.json', 'w') as f:
#     json.dump(config, f)


###--------------------------Make some plots--------------------------###

# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plt.plot(log['losses'])
plt.xlabel('Training Cycles', fontdict=font)
plt.ylabel('CE loss', fontdict=font)
plt.legend()
# plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
# plt.ylim([0.0, 1.0])
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
plt.show()

###--------------------------Run network after training for analysis--------------------------###

# """Run trained networks for analysis.

# Args:
#     envid: str, Environment ID

# Returns:
#     activity: a list of activity matrices, each matrix has shape (
#     N_time, N_neuron)
#     info: pandas dataframe, each row is information of a trial
#     config: dict of network, training configurations
# """

# def infer_test_timing(env):
#     """Infer timing of environment for testing."""
#     timing = {}
#     for period in env.timing.keys():
#         period_times = [env.sample_time(period) for _ in range(100)]
#         timing[period] = np.median(period_times)
#     return timing


# modelpath = get_modelpath(envid)
# with open(modelpath / 'config.json') as f:
#     config = json.load(f)

# env_kwargs = config['env_kwargs']

# # Run network to get activity and info
# # Environment
# env = gym.make(envid, **env_kwargs)
# env.timing = infer_test_timing(env)
# env.reset(no_step=True)

# # Instantiate the network and print information
# with torch.no_grad():
#     net = Net(input_size=env.observation_space.shape[0],
#               hidden_size=config['hidden_size'],
#               output_size=env.action_space.n)
#     net = net.to(device)
#     net.load_state_dict(torch.load(modelpath / 'net.pth'))

#     perf = 0
#     num_trial = 100

#     activity = list()
#     info = pd.DataFrame()

#     for i in range(num_trial):
#         env.new_trial()
#         ob, gt = env.ob, env.gt
#         inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
#         action_pred, hidden = net(inputs)

#         # Compute performance
#         action_pred = action_pred.detach().numpy()
#         choice = np.argmax(action_pred[-1, 0, :])
#         correct = choice == gt[-1]

#         # Log trial info
#         trial_info = env.trial
#         trial_info.update({'correct': correct, 'choice': choice})
#         info = info.append(trial_info, ignore_index=True)

#         # Log stimulus period activity
#         activity.append(np.array(hidden)[:, 0, :])

#     print('Average performance', np.mean(info['correct']))

# activity = np.array(activity)

###--------------------------General analysis--------------------------###
# def analysis_average_activity(activity, info, config):
#     # Load and preprocess results
#     plt.figure(figsize=(1.2, 0.8))
#     t_plot = np.arange(activity.shape[1]) * config['dt']
#     plt.plot(t_plot, activity.mean(axis=0).mean(axis=-1))

# analysis_average_activity(activity, info, config)


# def get_conditions(info):
#     """Get a list of task conditions to plot."""
#     conditions = info.columns
#     # This condition's unique value should be less than 5
#     new_conditions = list()
#     for c in conditions:
#         try:
#             n_cond = len(pd.unique(info[c]))
#             if 1 < n_cond < 5:
#                 new_conditions.append(c)
#         except TypeError:
#             pass
        
#     return new_conditions

# def analysis_activity_by_condition(activity, info, config):
#     conditions = get_conditions(info)
#     for condition in conditions:
#         values = pd.unique(info[condition])
#         plt.figure(figsize=(1.2, 0.8))
#         t_plot = np.arange(activity.shape[1]) * config['dt']
#         for value in values:
#             a = activity[info[condition] == value]
#             plt.plot(t_plot, a.mean(axis=0).mean(axis=-1), label=str(value))
#         plt.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))

# analysis_activity_by_condition(activity, info, config)


# def analysis_example_units_by_condition(activity, info, config):
#     conditions = get_conditions(info)
#     if len(conditions) < 1:
#         return

#     example_ids = np.array([0, 1])    
#     for example_id in example_ids:        
#         example_activity = activity[:, :, example_id]
#         fig, axes = plt.subplots(
#                 len(conditions), 1,  figsize=(1.2, 0.8 * len(conditions)),
#                 sharex=True)
#         for i, condition in enumerate(conditions):
#             ax = axes[i]
#             values = pd.unique(info[condition])
#             t_plot = np.arange(activity.shape[1]) * config['dt']
#             for value in values:
#                 a = example_activity[info[condition] == value]
#                 ax.plot(t_plot, a.mean(axis=0), label=str(value))
#             ax.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
#             ax.set_ylabel('Activity')
#             if i == len(conditions) - 1:
#                 ax.set_xlabel('Time (ms)')
#             if i == 0:
#                 ax.set_title('Unit {:d}'.format(example_id + 1))

# analysis_example_units_by_condition(activity, info, config)


# def analysis_pca_by_condition(activity, info, config):
#     # Reshape activity to (N_trial x N_time, N_neuron)
#     activity_reshape = np.reshape(activity, (-1, activity.shape[-1]))
#     pca = PCA(n_components=2)
#     pca.fit(activity_reshape)
    
#     conditions = get_conditions(info)
#     for condition in conditions:
#         values = pd.unique(info[condition])
#         fig = plt.figure(figsize=(2.5, 2.5))
#         ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
#         for value in values:
#             # Get relevant trials, and average across them
#             a = activity[info[condition] == value].mean(axis=0)
#             a = pca.transform(a)  # (N_time, N_PC)
#             plt.plot(a[:, 0], a[:, 1], label=str(value))
#         plt.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
    
#         plt.xlabel('PC 1')
#         plt.ylabel('PC 2')
