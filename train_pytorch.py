# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:55:15 2021

@author: weilong
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_task import RikhyeTaskBatch
import sys
sys.path.append('..')
from temp_model import MD, PytorchPFC

num_cueingcontext = 2
num_cue = 2
num_rule = 2
rule = [0, 1, 0, 1]
blocklen = [500, 500, 200]
block_cueingcontext = [0, 1, 0]
tsteps = 200
cuesteps = 100
batch_size = 1 # always set to 1 right now

dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule, rule=rule, blocklen=blocklen, \
block_cueingcontext=block_cueingcontext, tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)

