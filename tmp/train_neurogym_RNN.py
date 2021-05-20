import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from pathlib import Path

import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule



# task names in the Yang19 collection
#   ngym.get_collection('yang19') is defined in neurogym\envs\collections\__init__.py
#   'yang19.go-v0',         'yang19.rtgo-v0',   'yang19.dlygo-v0',  'yang19.anti-v0',      'yang19.rtanti-v0',
#   'yang19.dlyanti-v0',    'yang19.dm1-v0',    'yang19.dm2-v0',    'yang19.ctxdm1-v0',    'yang19.ctxdm2-v0',
#   'yang19.multidm-v0',    'yang19.dlydm1-v0', 'yang19.dlydm2-v0', 'yang19.ctxdlydm1-v0', 'yang19.ctxdlydm2-v0',
#   'yang19.multidlydm-v0', 'yang19.dms-v0',    'yang19.dnms-v0',   'yang19.dmc-v0',       'yang19.dnmc-v0'


# why gym.make can make Yang19 tasks?
#   because all tasks in the collection is registered in neurogym\envs\registration.py
#   once tasks are registered, gym.make works


# what task should we test in the next trial?
#   RandomSchedule helps us randomly choose a task
#   RandomSchedule is defined in neurogym\utils\scheduler.py


# final dimensions 
#   input size:  (53,)
#   output size:  17
#   inputs.shape:  torch.Size([200, 1, 53])
#   labels.shape:  torch.Size([200])
#   outputs.shape:  torch.Size([200, 1, 17])


# input dimensions
#   original input size is (33,) and action size is 17
#   ScheduleEnvs adds rule inputs to the observation and choose a task for the next trial based on the schedule
#   ScheduleEnvs is defined in neurogym\neurogym\wrappers\block.py
# meanings of input dimensions
#   index 0 is fixation input;
#   index 1-16 is the first input modality;
#   index 17-32 is the second input modality;
#   index 33-52 is the rule inputs indicating current environment
# tasks have either one or two input modality(modalities)
#   one input modality: 
#     yang19.go-v0
#     yang19.rtgo-v0
#     yang19.dlygo-v0
#     yang19.anti-v0
#     yang19.rtanti-v0
#     yang19.dlyanti-v0
#     yang19.dms-v0
#     yang19.dnms-v0
#     yang19.dmc-v0
#     yang19.dnmc-v0
#   two input modalities
#     yang19.dm1-v0
#     yang19.dm2-v0
#     yang19.ctxdm1-v0
#     yang19.ctxdm2-v0
#     yang19.multidm-v0
#     yang19.dlydm1-v0
#     yang19.dlydm2-v0
#     yang19.ctxdlydm1-v0
#     yang19.ctxdlydm2-v0
#     yang19.multidlydm-v0
#  for tasks that have only one input modality, the second input modality (index 17-32) of inputs are zero.
#  for tasks that have two input modalities, the econd input modality (index 17-32) of inputs are non-zero.


# meanings of output dimensions
#   index 0 is fixation output;
#   index 1-16 is the output modality (only one modality);
# when period == fixation, label = 0, only fixation output is expected to be activated
# when period == choice, label = ground truth, only the ground truth neuron is expected to be activated


# in neurogym\envs\collections\yang19.py
#   20 tasks/environments in the collection are defined 
#   env.observation_space.name of different environments is defined
#   env.action_space.name of different environments is defined
#   this file tells how we get observation and groundtruth.
# in neurogym\core.py.
#   TrialEnv class is defined
#   TrialEnv.add_ob is how we update observation in a new trial.



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

tasks = ngym.get_collection('yang19')
# check task names
# print(tasks)

envs = [gym.make(task, **config['env_kwargs']) for task in tasks]
# check original input and output shapes
# for i in range(len(envs)):
#     print(envs[i].observation_space.shape)
#     print(envs[i].action_space.n)
# ob_size are all (33,)
# act_size are all 17

# check input modalities
# for i in range(len(envs)):
#     print(tasks[i])
#     print(envs[i].observation_space.name['fixation'])
#     if 'stimulus' in envs[i].observation_space.name.keys():
#         print(envs[i].observation_space.name['stimulus'])
#     else:
#         print(envs[i].observation_space.name['stimulus_mod1'])
#         print(envs[i].observation_space.name['stimulus_mod2'])

# check output modalities
# for i in range(len(envs)):
#     print(tasks[i])
#     print(envs[i].action_space.name['fixation'])
#     print(envs[i].action_space.name['choice'])

schedule = RandomSchedule(len(envs))
# check how RandomSchedule works
# for _ in range(20):
#     print(schedule(), end=' ')
# output: 3 2 18 4 10 19 2 11 4 14 13 5 16 11 15 3 8 15 6 19

env = ScheduleEnvs(envs, schedule=schedule, env_input=True)

dataset = ngym.Dataset(env, batch_size=config['batch_size'], seq_len=config['seq_len'])
env = dataset.env

ob_size = env.observation_space.shape[0]
act_size = env.action_space.n


###--------------------------Define model--------------------------###

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out, hidden = self.lstm(x)
        out, hidden = self.rnn(x)
        x = self.linear(out)
        return x, out


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

net = Net(input_size  = config['input_size' ],
          hidden_size = config['hidden_size'],
          output_size = config['output_size'])
net = net.to(device)
print(net, '\n')

criterion = nn.CrossEntropyLoss()

print('training parameters:')
training_params = list()
for name, param in net.named_parameters():
    # if 'rnn' not in name:
    if True:
        print(name)
        training_params.append(param)
print()
optimizer = torch.optim.Adam(training_params, lr=config['lr'])


total_training_cycle = 40000
print_training_cycle = 100
running_loss = 0.0
running_train_time = 0
losses = list()


for i in range(total_training_cycle):

    train_time_start = time.time()

    inputs, labels = dataset()
    # check the task we are testing
    # print(tasks[dataset.env.i_env])
    # check the meaning of labels
    # inputs, labels = dataset()
    # print(inputs.shape, labels.shape)
    # print(tasks[dataset.env.i_env])
    # for i in range(20):
    #     print(inputs[i, 0, :], labels[i])

    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
    break

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs, _ = net(inputs)

    # check data dimensions
    # print("input size: ", env.observation_space.shape)
    # print("output size: ", env.action_space.n)
    # print("inputs.shape: ", inputs.shape)
    # print("labels.shape: ", labels.shape)
    # print("outputs.shape: ", outputs.shape)

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


# def get_modelpath(envid):
#     # Make local file directories
#     path = Path('.') / 'files'
#     os.makedirs(path, exist_ok=True)
#     path = path / envid
#     os.makedirs(path, exist_ok=True)
#     return path

# modelpath = get_modelpath(envid)

# save config
# with open(modelpath / 'config.json', 'w') as f:
#     json.dump(config, f)

# save model
# torch.save(net.state_dict(), modelpath / 'net.pth')


###--------------------------Analysis--------------------------###

# Cross Entropy loss
font = {'family':'Times New Roman','weight':'normal', 'size':30}
plt.plot(losses)
plt.xlabel('Training Cycles', fontdict=font)
plt.ylabel('CE loss', fontdict=font)
plt.legend()
# plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
# plt.ylim([0.0, 1.0])
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.tight_layout()
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
