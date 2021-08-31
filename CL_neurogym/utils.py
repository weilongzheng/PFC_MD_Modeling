import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_taskpairs():
    pass

def save_parameters():
    pass

def get_optimizer(net, config):
    print('training parameters:')
    training_params = list()
    for name, param in net.named_parameters():
        # if 'rnn.h2h' not in name: # reservoir
        # if True: # learnable RNN
        if 'rnn.input2PFCctx' not in name:
            print(name)
            training_params.append(param)
    print()
    optimizer = torch.optim.Adam(training_params, lr=config.lr)
    return optimizer

