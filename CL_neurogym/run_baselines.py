import os
import sys
from pathlib import Path
import json
import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym
import matplotlib.pyplot as plt
import seaborn as sns

from configs import get_config
from logger.logger import BaseLogger
from data import get_dataset
from models.PFC import RNN
from models import get_model
from utils import set_seed, get_optimizer, test_in_training
from analysis.visualization import plot_rnn_activity, plot_loss, plot_perf, plot_fullperf


# configs
mode = 'EWC'
config = get_config(mode)
print(mode)

# datasets
dataset = get_dataset(dataset_filename='ngym', config=config)

# set random seed
set_seed(seed=config.RNGSEED)

# backbone network
net = RNN(input_size=config.input_size,
          hidden_size=config.hidden_size,
          output_size=config.output_size)
net = net.to(config.device)
print(net, '\n')

# criterion & optimizer
criterion = nn.MSELoss()
optimizer, training_params, named_training_params = get_optimizer(net=net, config=config)

# continual learning model
CL_model = get_model(backbone=net,
                     loss=criterion,
                     config=config,
                     transform=None,
                     opt=optimizer,
                     device=config.device,
                     parameters=training_params,
                     named_parameters=named_training_params)
net = CL_model.net