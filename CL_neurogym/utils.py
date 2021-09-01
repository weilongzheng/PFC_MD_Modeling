import itertools
import numpy as np
import random
import torch
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_task_seqs():
    '''
    Generate task sequences
    '''
    ## 1. all pairs
    # tasks = ['yang19.dms-v0',
    #          'yang19.dnms-v0',
    #          'yang19.dmc-v0',
    #          'yang19.dnmc-v0',
    #          'yang19.dm1-v0',
    #          'yang19.dm2-v0',
    #          'yang19.ctxdm1-v0',
    #          'yang19.ctxdm2-v0',
    #          'yang19.multidm-v0',
    #          'yang19.dlygo-v0',
    #          'yang19.dlyanti-v0',
    #          'yang19.go-v0',
    #          'yang19.anti-v0',
    #          'yang19.rtgo-v0',
    #          'yang19.rtanti-v0']
    # task_seqs = list(itertools.permutations(tasks, 2))
    # task_seqs = [val for val in task_seqs for i in range(2)]
    ## 2. pairs from different task families
    GoFamily = ['yang19.dlygo-v0', 'yang19.dlyanti-v0']
    DMFamily = ['yang19.dm1-v0', 'yang19.ctxdm2-v0', 'yang19.multidm-v0']
    MatchFamily = ['yang19.dms-v0', 'yang19.dmc-v0', 'yang19.dnms-v0', 'yang19.dnmc-v0']
    TaskA = GoFamily + DMFamily
    TaskB = MatchFamily
    task_seqs = []
    for a in TaskA:
        for b in TaskB:
            task_seqs.append((a, b))
            task_seqs.append((b, a))
    return task_seqs

# training
def get_optimizer(net, config):
    print('training parameters:')
    training_params = list()
    named_training_params = dict()
    for name, param in net.named_parameters():
        # if 'rnn.h2h' not in name: # reservoir
        # if True: # learnable RNN
        if 'rnn.input2PFCctx' not in name:
            print(name)
            training_params.append(param)
            named_training_params[name] = param
    optimizer = torch.optim.Adam(training_params, lr=config.lr)
    return optimizer, training_params, named_training_params

def forward_backward(net, opt, crit, inputs, labels, task_id):
    '''
    forward + backward + optimize
    '''
    opt.zero_grad()
    outputs, rnn_activity = net(inputs, task_id=task_id)
    loss = crit(outputs, labels)
    loss.backward()
    opt.step()
    return loss, rnn_activity

# testing
# get_performance is not used
def get_performance(net, dataset, task_id, config):
    perf = 0
    for i in range(num_trial):
        ob, gt = dataset.new_trial(task_id)

        inputs = torch.from_numpy(ob).type(torch.float).to(config.device)
        action_pred, _ = net(inputs, task_id=task_id)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]

    perf /= num_trial
    return perf

def get_full_performance(net, dataset, task_id, config):
    num_trial = config.test_num_trials
    fix_perf = 0.
    act_perf = 0.
    num_no_act_trial = 0
    for i in range(num_trial):
        ob, gt = dataset.new_trial(task_id)
        inputs = torch.from_numpy(ob).type(torch.float).to(config.device)
        inputs = inputs[:, np.newaxis, :]
        action_pred, _ = net(inputs, task_id=task_id)
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

def test_in_training(net, dataset, config, log, trial_idx):
    '''
    Compute performances for every task in the given task sequence(config.task_seq).
    '''
    # turn on test mode
    net.eval()
    if hasattr(config, 'MDeffect'):
        if config.MDeffect:
            net.rnn.md.learn = False
    # testing
    with torch.no_grad():
        log.stamps.append(trial_idx+1)
        #   fixation & action performance
        print('Performance')
        for env_id in range(config.num_task):
            fix_perf, act_perf = get_full_performance(net=net, dataset=dataset, task_id=env_id, config=config)
            log.fix_perfs[env_id].append(fix_perf)
            log.act_perfs[env_id].append(act_perf)
            print('  fix performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, trial_idx+1, fix_perf))
            print('  act performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, trial_idx+1, act_perf))
    # back to train mode
    net.train()
    if hasattr(config, 'MDeffect'):
        if config.MDeffect:
            net.rnn.md.learn = True

# save variables
def save_parameters():
    pass