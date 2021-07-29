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


###--------------------------Dataset configs--------------------------###
config = {
    'RNGSEED': 5,
    'env_kwargs': {'dt': 100},
    'tasks': ngym.get_collection('yang19'),
}

# set random seed
RNGSEED = config['RNGSEED']
random.seed(RNGSEED)
np.random.seed(RNGSEED)
torch.manual_seed(RNGSEED)


###--------------------------Generate dataset--------------------------###
tasks = config['tasks']
print(tasks)

for task in tasks:
    env = gym.make(task, **config['env_kwargs'])
    env.new_trial()
    ob, gt = env.ob, env.gt # ob:(trial_len, input_size); gt:(trial_len)

    font = {'family':'Times New Roman','weight':'normal', 'size':12}
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(task, fontsize=20)
    ax1 = plt.subplot(121)
    ax1 = sns.heatmap(ob.T, cmap='Reds')
    ax1.set_yticklabels([0]+[i for i in range(1, 17)]*2, rotation=0)
    ax1.set_xlabel('Timestep', fontdict=font)
    ax1.set_ylabel('Input dim', fontdict=font)
    ax2 = plt.subplot(122)
    plt.plot(gt.T)
    plt.xlabel('Timestep', fontdict=font)
    plt.ylabel('Inputs', fontdict=font)
    plt.tight_layout()
    plt.savefig('./files/' + 'obgt_' + task + '.png')
    plt.close()