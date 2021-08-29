import itertools
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule

###--------------------------Helper functions--------------------------###

def get_full_performance(net, env, task_id, num_task, num_trial=1000, device='cpu'):
    fix_perf = 0.
    act_perf = 0.
    num_no_act_trial = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob[:, 1:] = (ob[:, 1:] - np.min(ob[:, 1:]))/(np.max(ob[:, 1:]) - np.min(ob[:, 1:]))
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred, _ = net(inputs, sub_id=task_id)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)

        fix_len = sum(gt == 0)
        act_len = len(gt) - fix_len
        assert all(gt[:fix_len] == 0)
        fix_perf += sum(action_pred[:fix_len, 0] == 0)/fix_len
        if act_len != 0:
            assert all(gt[fix_len:] == gt[-1])
            # act_perf += sum(action_pred[fix_len:, 0] == gt[-1])/act_len
            act_perf += (action_pred[-1, 0] == gt[-1])
        else: # no action in this trial
            num_no_act_trial += 1

    fix_perf /= num_trial
    act_perf /= num_trial - num_no_act_trial
    return fix_perf, act_perf

def get_task_pair_id(task_pair):
    task_pair = tuple(task_pair)
    tasks = ['yang19.dms-v0',
             'yang19.dnms-v0',
             'yang19.dmc-v0',
             'yang19.dnmc-v0',
             'yang19.dm1-v0',
             'yang19.dm2-v0',
             'yang19.ctxdm1-v0',
             'yang19.ctxdm2-v0',
             'yang19.multidm-v0',
             'yang19.dlygo-v0',
             'yang19.dlyanti-v0',
             'yang19.go-v0',
             'yang19.anti-v0',
             'yang19.rtgo-v0',
             'yang19.rtanti-v0']
    num_tasks = 2
    task_pairs = list(itertools.permutations(tasks, num_tasks))
    task_pairs = [val for val in task_pairs for i in range(2)]
    for task_pair_id in range(len(task_pairs)):
        if task_pairs[task_pair_id] == task_pair:
            return task_pair_id
