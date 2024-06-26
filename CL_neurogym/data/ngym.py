import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym

class NGYM():
    def __init__(self, config):
        self.config = config
        self.task_seq = tuple(self.config.task_seq)
        self.num_tasks = len(self.task_seq)
        # initialize environments
        self.envs = []
        for task in self.task_seq:
            env = gym.make(task, **self.config.env_kwargs)
            self.envs.append(env)

    def new_trial(self, task_id):
        # fetch data
        env = self.envs[task_id]
        env.new_trial()
        ob, gt = env.ob, env.gt
        assert not np.any(np.isnan(ob))
        assert not np.any(np.isnan(gt))
        # preprocessing
        ob[:, 1:] = (ob[:, 1:] - np.min(ob[:, 1:]))/(np.max(ob[:, 1:]) - np.min(ob[:, 1:]))
        return ob, gt

    def __call__(self, task_id):
        ob, gt = self.new_trial(task_id)
        # numpy -> torch
        inputs = torch.from_numpy(ob).type(torch.float).to(self.config.device)
        labels = torch.from_numpy(gt).type(torch.long).to(self.config.device)
        # index -> one-hot vector
        labels = (F.one_hot(labels, num_classes=self.config.output_size)).float()
        # add batch axis
        inputs = inputs[:, np.newaxis, :]
        labels = labels[:, np.newaxis, :]
        return inputs, labels
