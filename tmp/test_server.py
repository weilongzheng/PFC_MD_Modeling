import os
import sys
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

print('hello world!')