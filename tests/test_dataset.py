'''
Test dataset
'''

import pytest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from task import RikhyeTask, RikhyeTaskBatch


def test_RikhyeTaskBatch():
    '''
    Test RikhyeTaskBatch class if inputs match labels
    '''
    RNGSEED = 5 # set random seed; default 5
    np.random.seed([RNGSEED])
    torch.manual_seed(RNGSEED)
    os.environ['PYTHONHASHSEED'] = str(RNGSEED)

    #---------------- Dataset ----------------#

    # Generate RikhyeTaskBatch trainset
    num_cueingcontext = 2
    num_cue = 2
    num_rule = 2
    rule = [0, 1, 0, 1]
    blocklen = [200, 200, 200]
    block_cueingcontext = [0, 1, 0]
    tsteps = 200
    cuesteps = 100
    batch_size = 1 # always 1

    dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule,\
                            rule=rule, blocklen=blocklen, block_cueingcontext=block_cueingcontext,\
                            tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)

    #---------------- Test ----------------#
    total_step = sum(blocklen)//batch_size

    for i in range(total_step):
        inputs, labels = dataset()
        assert inputs.shape[0] == labels.shape[0] # seq len
        assert inputs.shape[1] == labels.shape[1] # batch size
        inputs = inputs[:, 0, :]
        labels = labels[:, 0, :]

        # reconstruct labels
        seq_len = inputs.shape[0]
        expected_labels = np.zeros_like(labels)
        for t in range(seq_len):
            idx = np.where(inputs[t, :] == 1)[0]
            if len(idx) == 0:
                # delay period always comes after cueing period
                expected_labels[t, :] = expected_label
            else:
                # cueing period
                idx = int(idx) # numpy array -> int
                expected_label = np.zeros_like(labels[t, :])
                expected_label[rule[idx]] = 1
                expected_labels[t, :] = expected_label

        # Compare expected labels with labels
        assert (expected_labels == labels).all() == True

