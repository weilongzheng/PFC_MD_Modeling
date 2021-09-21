import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


# scale up test performance
# TODO: make this a helper function
## compute mean & std of performance
if 0:
    # FILE_PATH = './files/scaleup_threetasks/baselines/'
    FILE_PATH = './files/scaleup_threetasks_3/noisestd0dot01/'

    # settings = ['EWC', 'SI', 'PFC']
    settings = ['PFCMD']

    ITER = list(range(36))
    LEN = len(ITER)
    for setting in settings:
        act_perfs_all = []
        for i in ITER:
            PATH = FILE_PATH + str(i) + '_log_' + setting + '.npy'
            log = np.load(PATH, allow_pickle=True).item()
            act_perfs_all.append(np.array(log.act_perfs))
        act_perfs_all = np.stack(act_perfs_all, axis=0)
        time_stamps = log.stamps
        act_perfs_mean = np.mean(act_perfs_all, axis=0)
        act_perfs_std = np.std(act_perfs_all, axis=0)
        np.save('./files/'+'avg_perfs_mean_'+setting+'noisestd0dot01.npy', act_perfs_mean)
        np.save('./files/'+'avg_perfs_std_'+setting+'noisestd0dot01.npy', act_perfs_std)
        np.save('./files/'+'time_stamps_'+setting+'noisestd0dot01.npy', time_stamps)

# main performance curve: two tasks
if 0:
    FILE_PATH = './files/scaleup_twotasks_4/'
    setting = 'PFCMDnoisestd0dot01'
    act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + setting + '.npy')
    act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + setting + '.npy')
    time_stamps = np.load(FILE_PATH + 'time_stamps_' + setting + '.npy')

    fig, axes = plt.subplots(figsize=(6, 4))
    ax = axes
    line_colors = ['tab:red', 'tab:blue']
    labels = ['Task1', 'Task2']
    linewidth = 2
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    ax.axvspan(    0, 20000, alpha=0.08, color=line_colors[0])
    ax.axvspan(20000, 40000, alpha=0.08, color=line_colors[1])
    ax.axvspan(40000, 50000, alpha=0.08, color=line_colors[0])

    for env_id in range(2): # 2 tasks
        plt.plot(time_stamps, act_perfs_mean[env_id, :],
                 linewidth=linewidth, color=line_colors[env_id], label=labels[env_id])
        plt.fill_between(time_stamps, 
                         act_perfs_mean[env_id, :]-act_perfs_std[env_id, :],
                         act_perfs_mean[env_id, :]+act_perfs_std[env_id, :],
                         alpha=0.2, color=line_colors[env_id])
    plt.xlabel('Trials', fontdict=label_font)
    plt.ylabel('Performance', fontdict=label_font)
    plt.title('Task Performance', fontdict=title_font)
    plt.xlim([0.0, 51000])
    plt.ylim([0.0, 1.01])
    plt.yticks([0.1*i for i in range(11)])
    plt.legend(bbox_to_anchor=(1.0, 0.65), prop=legend_font)
    plt.tight_layout()
    plt.show()
    # plt.savefig(FILE_PATH + 'performance.pdf')
    # plt.close()

# main performance curve: three tasks
if 0:
    FILE_PATH = './files/scaleup_threetasks_3/'
    setting = 'PFCMDnoisestd0dot01'
    act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + setting + '.npy')
    act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + setting + '.npy')
    time_stamps = np.load(FILE_PATH + 'time_stamps_' + setting + '.npy')

    fig, axes = plt.subplots(figsize=(6, 4))
    ax = axes
    line_colors = ['tab:red', 'tab:blue', 'tab:green']
    labels = ['Task1', 'Task2', 'Task3']
    linewidth = 2
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    ax.axvspan(    0, 20000, alpha=0.08, color=line_colors[0])
    ax.axvspan(20000, 40000, alpha=0.08, color=line_colors[1])
    ax.axvspan(40000, 60000, alpha=0.08, color=line_colors[2])
    ax.axvspan(60000, 70000, alpha=0.08, color=line_colors[0])

    for env_id in range(3): # 3 tasks
        plt.plot(time_stamps, act_perfs_mean[env_id, :],
                 linewidth=linewidth, color=line_colors[env_id], label=labels[env_id])
        plt.fill_between(time_stamps, 
                         act_perfs_mean[env_id, :]-act_perfs_std[env_id, :],
                         act_perfs_mean[env_id, :]+act_perfs_std[env_id, :],
                         alpha=0.2, color=line_colors[env_id])
    plt.xlabel('Trials', fontdict=label_font)
    plt.ylabel('Performance', fontdict=label_font)
    plt.title('Task Performance', fontdict=title_font)
    plt.xlim([0.0, 71000])
    plt.ylim([0.0, 1.01])
    plt.yticks([0.1*i for i in range(11)])
    plt.legend(bbox_to_anchor=(1.0, 0.65), prop=legend_font)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FILE_PATH + 'performance.pdf')
    plt.close()

# PFC+MD VS baselines: two tasks
if 0:
    FILE_PATH = './files/scaleup_twotasks_4/'
    settings = ['PFCMDnoisestd0dot01', 'EWC', 'SI', 'PFC']
    line_colors = ['tab:red', 'darkviolet', 'darkgreen', 'black']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC+SI', 'PFC']
    linewidths = [3, 2, 2, 2]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for env_id in range(2): # 2 tasks
        color1, color2= 'tab:red', 'tab:blue'
        axes[env_id].axvspan(    0, 20000, alpha=0.1, color=color1)
        axes[env_id].axvspan(20000, 40000, alpha=0.1, color=color2)
        axes[env_id].axvspan(40000, 50000, alpha=0.1, color=color1)
        for i in range(len(settings)):
            act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + settings[i] + '.npy')
            act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + settings[i] + '.npy')
            time_stamps = np.load(FILE_PATH + 'time_stamps_' + settings[i] + '.npy')
            axes[env_id].plot(time_stamps, act_perfs_mean[env_id, :], linewidth=linewidths[i], color=line_colors[i], label=labels[i])
            axes[env_id].set_xlabel('Trials', fontdict=label_font)
            axes[env_id].set_ylabel('Performance', fontdict=label_font)
            axes[env_id].set_title('Task{:d} Performance'.format(env_id+1), fontdict=title_font)
            axes[env_id].set_xlim([0.0, 51000])
            axes[env_id].set_ylim([0.0, 1.01])
            axes[env_id].set_yticks([0.1*i for i in range(11)])
    axes[-1].legend(bbox_to_anchor = (1.0, 0.65), prop=legend_font)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FILE_PATH + 'performance{:d}.pdf'.format(env_id+1))
    plt.close()

# PFC+MD VS baselines: three tasks
if 0:
    FILE_PATH = './files/scaleup_threetasks_3/'
    settings = ['PFCMDnoisestd0dot01', 'EWC', 'SI', 'PFC']
    line_colors = ['tab:red', 'darkviolet', 'darkgreen', 'black']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC+SI', 'PFC']
    linewidths = [3, 2, 2, 2]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for env_id in range(3): # 2 tasks
        color1, color2, color3= 'tab:red', 'tab:blue', 'tab:green'
        axes[env_id].axvspan(    0, 20000, alpha=0.1, color=color1)
        axes[env_id].axvspan(20000, 40000, alpha=0.1, color=color2)
        axes[env_id].axvspan(40000, 60000, alpha=0.1, color=color3)
        axes[env_id].axvspan(60000, 70000, alpha=0.1, color=color1)
        for i in range(len(settings)):
            act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + settings[i] + '.npy')
            act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + settings[i] + '.npy')
            time_stamps = np.load(FILE_PATH + 'time_stamps_' + settings[i] + '.npy')
            axes[env_id].plot(time_stamps, act_perfs_mean[env_id, :], linewidth=linewidths[i], color=line_colors[i], label=labels[i])
            axes[env_id].set_xlabel('Trials', fontdict=label_font)
            axes[env_id].set_ylabel('Performance', fontdict=label_font)
            axes[env_id].set_title('Task{:d} Performance'.format(env_id+1), fontdict=title_font)
            axes[env_id].set_xlim([0.0, 71000])
            axes[env_id].set_ylim([0.0, 1.01])
            axes[env_id].set_yticks([0.1*i for i in range(11)])
    axes[-1].legend(bbox_to_anchor = (1.0, 0.65), prop=legend_font)
    plt.tight_layout()
    plt.show()
    # plt.savefig(FILE_PATH + 'performance{:d}.pdf'.format(env_id+1))
    # plt.close()

# Parametric noise
if 0:
    FILE_PATH = './files/scaleup_twotasks_4/'
    settings = ['PFCMDnoisestd0dot01', 'PFCMDnoisestd0dot1']
    line_colors = ['tab:red', 'tab:blue']
    labels = ['$\sigma_{noise}=0.01$', '$\sigma_{noise}=0.1$']
    linewidths = [2, 2]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for env_id in range(2): # 2 tasks
        color1, color2= 'tab:red', 'tab:blue'
        axes[env_id].axvspan(    0, 20000, alpha=0.08, color=color1)
        axes[env_id].axvspan(20000, 40000, alpha=0.08, color=color2)
        axes[env_id].axvspan(40000, 50000, alpha=0.08, color=color1)
        for i in range(len(settings)):
            act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + settings[i] + '.npy')
            act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + settings[i] + '.npy')
            time_stamps = np.load(FILE_PATH + 'time_stamps_' + settings[i] + '.npy')
            axes[env_id].plot(time_stamps, act_perfs_mean[env_id, :],
                     linewidth=linewidths[i], color=line_colors[i], label=labels[i])
            axes[env_id].fill_between(time_stamps, 
                             act_perfs_mean[env_id, :]-act_perfs_std[env_id, :],
                             act_perfs_mean[env_id, :]+act_perfs_std[env_id, :],
                             alpha=0.2, color=line_colors[i])
            axes[env_id].set_xlabel('Trials', fontdict=label_font)
            axes[env_id].set_ylabel('Performance', fontdict=label_font)
            axes[env_id].set_title('Task{:d} Performance'.format(env_id+1), fontdict=title_font)
            axes[env_id].set_xlim([0.0, 51000])
            axes[env_id].set_ylim([0.0, 1.01])
            axes[env_id].set_yticks([0.1*i for i in range(11)])
    axes[-1].legend(bbox_to_anchor = (1.0, 0.65), prop=legend_font)
    plt.tight_layout()
    # plt.show()
    plt.savefig(FILE_PATH + 'performance{:d}.pdf'.format(env_id+1))
    plt.close()

# Trajectory
if 1:
    log = np.load('./files/trajectory/'+'log.npy', allow_pickle=True).item()
    config = np.load('./files/trajectory/'+'config.npy', allow_pickle=True).item()
    dataset = np.load('./files/trajectory/'+'dataset.npy', allow_pickle=True).item()
    net = torch.load('./files/trajectory/'+'net.pt')
    crit = nn.MSELoss()

    # turn on test mode
    net.eval()
    if hasattr(config, 'MDeffect'):
        if config.MDeffect:
            net.rnn.md.learn = False
    # testing
    with torch.no_grad():
        task_id = 0
        inputs, labels = dataset(task_id=task_id)
        outputs, rnn_activity = net(inputs, task_id=task_id)
        loss = crit(outputs, labels)
        print(loss)
