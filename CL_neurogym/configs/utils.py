import itertools
import numpy as np
import random
import torch
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym


# given task seq, return similarity
def get_similarity(task_seq):
    assert len(task_seq) == 2
    tick_names = np.load('./files/similarity/tick_names.npy')
    tick_names_dict = np.load('./files/similarity/tick_names_dict.npy', allow_pickle=True).item()
    tick_names_dict_reversed = np.load('./files/similarity/tick_names_dict_reversed.npy', allow_pickle=True).item()
    norm_similarity_matrix = np.load('./files/similarity/norm_similarity_matrix.npy')
    x = int(tick_names_dict_reversed[task_seq[0][len('yang19.'):-len('-v0')]])
    y = int(tick_names_dict_reversed[task_seq[1][len('yang19.'):-len('-v0')]])
    return norm_similarity_matrix[x, y]