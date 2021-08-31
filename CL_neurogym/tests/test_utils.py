import numpy as np
import random
import torch

def test_set_seed():
    from configs.configs import BaseConfig
    from utils import set_seed
    config = BaseConfig()
    for _ in range(10):
        set_seed(config.RNGSEED)
        print(np.random.rand(2), torch.rand(size=(2,)))

def test_get_taskpairs():
    from utils import get_taskpairs
    taskpairs = get_taskpairs()
    print(taskpairs)
