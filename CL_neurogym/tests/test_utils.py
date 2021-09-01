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

def test_get_task_seqs():
    from utils import get_task_seqs
    task_seqs = get_task_seqs()
    print(task_seqs)
