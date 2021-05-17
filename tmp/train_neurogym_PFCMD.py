import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
sys.path.append('D:\\DESKTOP\\Lab\\Projects\\Yang Lab\\neurogym') # directory of local neurogym module
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from pygifsicle import optimize

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import gym
import neurogym as ngym
from model_dev import PytorchPFCMD



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

envid_list = ['PerceptualDecisionMaking-v0', 'ContextDecisionMaking-v0', 'DelayMatchCategory-v0', 'DualDelayMatchSample-v0']
block1 = envid_list[0:2]
block2 = envid_list[2:4]
print('Training paradigm: block1 -> block2 -> block1')
print('Training task block1', block1)
print('Training task block2', block2, '\n')

config = {
    'RNGSEED': 5,
    'dt': 100,
    'hidden_size': 64,
    'lr': 1e-2,
    'batch_size': 1,
    'seq_len': 200,
    'envid_list': envid_list,
}

env_kwargs = {'dt': config['dt']}
config['env_kwargs'] = env_kwargs

# set random seed
RNGSEED = config['RNGSEED']
np.random.seed([RNGSEED])
torch.manual_seed(RNGSEED)


###--------------------------Generate dataset--------------------------###

# Supervised dataset list
dataset_block1 = []
for i in range(len(block1)):
    dataset = ngym.Dataset(block1[i], env_kwargs=env_kwargs, \
                                      batch_size=config['batch_size'], \
                                      seq_len=config['seq_len'])
    dataset_block1.append(dataset)
dataset_block2 = []
for i in range(len(block2)):
    dataset = ngym.Dataset(block2[i], env_kwargs=env_kwargs, \
                                      batch_size=config['batch_size'], \
                                      seq_len=config['seq_len'])
    dataset_block2.append(dataset)

# number of environments in one block
Nevns = len(dataset_block1)
# observation space
ob_size = sum([dataset_block1[i].env.observation_space.shape[0] for i in range(Nevns)])
# action space
act_size = [dataset_block1[i].env.action_space.n for i in range(Nevns)]
assert len(np.unique(act_size)) == 1 # the action sapces in the block should be the same
act_size = np.unique(act_size)[0]

# Get data from blocks
def get_data(dataset_list, ob_size, act_size, seq_len):
    Nevns = len(dataset_list)
    total_seq_len = Nevns*seq_len

    # transform ob_size_list: [1,2,3] -> [0,1,3,6]
    ob_size_list = [dataset_list[i].env.observation_space.shape[0] for i in range(Nevns)]
    ob_size_transformed = [0]
    for i in range(Nevns):
        ob_size_transformed.append(sum(ob_size_list[0:i+1]))
    
    # allocate memory
    inputs = np.zeros(shape=(total_seq_len, config['batch_size'], ob_size))

    # get data
    for i in range(Nevns):
        input, label = dataset_list[i]()
        inputs[seq_len*i:seq_len*(i+1), :, ob_size_transformed[i]:ob_size_transformed[i+1]] = input
        if i == 0:
            labels = label
        else:
            labels = np.concatenate((labels, label))

    return inputs, labels
    # return input, label, inputs, labels

# input, label, inputs, labels = get_data(dataset_list, ob_size, act_size, config['seq_len'])


###--------------------------Model configs--------------------------###

# Model settings
model_config = {
    'n_neuron': 1000,
    'n_neuron_per_cue': 200,
    'Num_MD': 10,
    'num_active': 5, # num MD active per context
    'n_output': act_size,
    'MDeffect': False,
    'PFClearn': False,
}
config.update(model_config)

###--------------------------Train network--------------------------###

model = PytorchPFCMD(Num_PFC=config['n_neuron'], n_neuron_per_cue=config['n_neuron_per_cue'], \
                     Num_MD=config['Num_MD'], num_active=config['num_active'], \
                     num_output=config['n_output'], MDeffect=config['MDeffect'])
print(model, '\n')

criterion = nn.CrossEntropyLoss()

print('training parameters:')
training_params = list()
for name, param in model.named_parameters():
    print(name)
    training_params.append(param)
if config['PFClearn']==True:
    print('pfc.Jrec')
    print('\n', end='')
    training_params.append(model.pfc.Jrec)
else:
    print('\n', end='')


total_training_cycle = 3000
print_training_cycle = 50
running_loss = 0.0
running_train_time = 0
losses = list()


for i in range(total_training_cycle):

    train_time_start = time.time()

    # inputs, labels = dataset()
    if i < 1000:
        inputs, labels = get_data(dataset_block1, ob_size, act_size, config['seq_len'])
    elif i >= 1000 and i < 2000:
        inputs, labels = get_data(dataset_block2, ob_size, act_size, config['seq_len'])
    elif i >= 2000:
        inputs, labels = get_data(dataset_block1, ob_size, act_size, config['seq_len'])
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs, _ = net(inputs)

    loss = criterion(outputs.view(-1, act_size), labels)
    loss.backward()
    optimizer.step()

    # print statistics
    losses.append(loss.item())
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
# torch.save(net.state_dict(), modelpath / 'net.pth')

# save config
# with open(modelpath / 'config.json', 'w') as f:
#     json.dump(config, f)


###--------------------------Make some plots--------------------------###

# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plt.plot(losses)
plt.xlabel('Training Cycles', fontdict=font)
plt.ylabel('CE loss', fontdict=font)
plt.legend()
plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
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
