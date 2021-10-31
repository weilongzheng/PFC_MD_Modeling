import os
import random
import numpy as np
import gym
import neurogym as ngym
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import get_task_seqs, get_task_seq_id
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# for the settings below, please check
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams["figure.titlesize"] = 20 # 'large'
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.labelpad'] = 7.0 # 4.0
mpl.rcParams['axes.linewidth'] = 1.6 # 0.8
mpl.rcParams['axes.labelsize'] = 15 # 'medium'
mpl.rcParams['axes.labelweight'] = 'normal'
mpl.rcParams['axes.titlesize'] = 20 # 'large'
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['xtick.labelsize'] = 12 # 'medium'
mpl.rcParams['ytick.labelsize'] = 12 # 'medium'
mpl.rcParams['xtick.major.width'] = 1.6 # 0.8
mpl.rcParams['xtick.minor.width'] = 1.2 # 0.6
mpl.rcParams['ytick.major.width'] = 1.6 # 0.8
mpl.rcParams['ytick.minor.width'] = 1.2 # 0.6
mpl.rcParams['xtick.major.size'] = 5.0 # 3.5
mpl.rcParams['xtick.minor.size'] = 4.0 # 2.0
mpl.rcParams['ytick.major.size'] = 5.0 # 3.5
mpl.rcParams['ytick.minor.size'] = 4.0 # 2.0
mpl.rcParams['legend.frameon'] = False # True
mpl.rcParams['legend.fontsize'] = 12 # 'medium'


# scale up test performance
# TODO: make this a helper function
## compute mean & std of performance
if 0:
    # FILE_PATH = './files/scaleup_threetasks_4/baselines/'
    # FILE_PATH = './files/scaleup_threetasks_4/PFCMD/'
    FILE_PATH = './files/randomortho_init/baselines/'
    # FILE_PATH = './files/randomortho_init/PFCMD/'

    settings = ['EWC', 'SI', 'PFC']
    # settings = ['PFCMD']

    ITER = list(range(420))
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
        np.save('./files/' + 'avg_perfs_mean_'+setting+'.npy', act_perfs_mean)
        np.save('./files/' + 'avg_perfs_std_'+setting+'.npy', act_perfs_std)
        np.save('./files/' + 'time_stamps_'+setting+'.npy', time_stamps)
    

# main performance curve: two tasks
if 0:
    FILE_PATH = './files/scaleup_twotasks_5/'
    # FILE_PATH = './files/randomortho_init/'
    setting = 'PFCMD'
    act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + setting + '.npy')
    act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + setting + '.npy')
    time_stamps = np.load(FILE_PATH + 'time_stamps_' + setting + '.npy')

    fig, axes = plt.subplots(figsize=(4, 4))
    ax = axes
    line_colors = ['tab:orange', 'tab:green']
    labels = ['Task1', 'Task2']
    linewidth = 2
    ax.axvspan(    0, 20000, alpha=0.1, color=line_colors[0])
    ax.axvspan(20000, 40000, alpha=0.1, color=line_colors[1])
    ax.axvspan(40000, 50000, alpha=0.1, color=line_colors[0])

    for env_id in range(2): # 2 tasks
        plt.plot(time_stamps, act_perfs_mean[env_id, :],
                 linewidth=linewidth, color=line_colors[env_id], label=labels[env_id])
        plt.fill_between(time_stamps, 
                         act_perfs_mean[env_id, :]-act_perfs_std[env_id, :],
                         act_perfs_mean[env_id, :]+act_perfs_std[env_id, :],
                         alpha=0.2, color=line_colors[env_id])
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    # plt.title('Task Performance')
    plt.xlim([0.0, 51000])
    plt.ylim([0.0, 1.01])
    plt.xticks(ticks=[0, 20000, 40000, 50000], labels=[0, 20, 40, 50])
    plt.yticks([0.2*i for i in range(6)])
    plt.legend(loc='lower right', bbox_to_anchor=(0.99, 0.05))
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'twotasksperformance-alltaskseqs.pdf')
    # plt.close()

# main performance curve: three tasks
if 0:
    FILE_PATH = './files/scaleup_threetasks_4/'
    setting = 'PFCMD'
    act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + setting + '.npy')
    act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + setting + '.npy')
    time_stamps = np.load(FILE_PATH + 'time_stamps_' + setting + '.npy')

    fig, axes = plt.subplots(figsize=(5, 4))
    ax = axes
    line_colors = ['tab:orange', 'tab:green', 'darkcyan']
    labels = ['Task1', 'Task2', 'Task3']
    linewidth = 2
    ax.axvspan(    0, 20000, alpha=0.1, color=line_colors[0])
    ax.axvspan(20000, 40000, alpha=0.1, color=line_colors[1])
    ax.axvspan(40000, 60000, alpha=0.1, color=line_colors[2])
    ax.axvspan(60000, 70000, alpha=0.1, color=line_colors[0])

    for env_id in range(3): # 3 tasks
        plt.plot(time_stamps, act_perfs_mean[env_id, :],
                 linewidth=linewidth, color=line_colors[env_id], label=labels[env_id])
        plt.fill_between(time_stamps, 
                         act_perfs_mean[env_id, :]-act_perfs_std[env_id, :],
                         act_perfs_mean[env_id, :]+act_perfs_std[env_id, :],
                         alpha=0.2, color=line_colors[env_id])
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    # plt.title('Task Performance')
    plt.xlim([0.0, 71000])
    plt.ylim([0.0, 1.01])
    plt.xticks(ticks=[0, 20000, 40000, 60000, 70000], labels=[0, 20, 40, 60, 70])
    plt.yticks([0.2*i for i in range(6)])
    plt.legend(loc='lower right', bbox_to_anchor=(1.00, 0.05))
    plt.tight_layout()
    # plt.show()
    plt.savefig('./files/' + 'threetasksperformance.pdf')
    plt.close()

# PFC+MD VS baselines: two tasks
if 0:
    FILE_PATH = './files/scaleup_twotasks_5/'
    # FILE_PATH = './files/randomortho_init/'
    settings = ['PFCMD', 'EWC', 'SI', 'PFC']
    line_colors = ['darkviolet', 'brown', 'tab:olive', 'darkgrey']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC+SI', 'PFC']
    linewidths = [2, 2, 2, 2]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for env_id in range(2): # 2 tasks
        color1, color2= 'tab:orange', 'tab:green'
        axes[env_id].axvspan(    0, 20000, alpha=0.15, color=color1)
        axes[env_id].axvspan(20000, 40000, alpha=0.15, color=color2)
        axes[env_id].axvspan(40000, 50000, alpha=0.15, color=color1)
        for i in range(len(settings)):
            act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + settings[i] + '.npy')
            act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + settings[i] + '.npy')
            time_stamps = np.load(FILE_PATH + 'time_stamps_' + settings[i] + '.npy')
            axes[env_id].plot(time_stamps, act_perfs_mean[env_id, :], linewidth=linewidths[i], color=line_colors[i], label=labels[i])
            axes[env_id].set_xlabel('Trials')
            axes[env_id].set_ylabel('Performance')
            # axes[env_id].set_title('Task{:d} Performance'.format(env_id+1))
            axes[env_id].set_xlim([0.0, 51000])
            axes[env_id].set_ylim([0.0, 1.01])
            axes[env_id].set_xticks([0, 20000, 40000, 50000])
            axes[env_id].set_yticks([0.2*i for i in range(6)])
            axes[env_id].set_xticklabels([0, 20, 40, 50])
            axes[env_id].set_yticklabels([round(0.2*i, 1) for i in range(6)])
            axes[env_id].legend(loc='lower right', bbox_to_anchor=(0.81, 0.05))
    plt.tight_layout()
    # plt.show()
    plt.savefig('./files/' + 'twotasksperformance-baselines-alltaskseqs.pdf')
    plt.close()

# PFC+MD VS baselines: three tasks
if 0:
    FILE_PATH = './files/scaleup_threetasks_4/'
    settings = ['PFCMD', 'EWC', 'SI', 'PFC']
    line_colors = ['darkviolet', 'brown', 'tab:olive', 'darkgrey']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC+SI', 'PFC']
    linewidths = [2, 2, 2, 2]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for env_id in range(3): # 2 tasks
        color1, color2, color3= 'tab:orange', 'tab:green', 'darkcyan'
        axes[env_id].axvspan(    0, 20000, alpha=0.1, color=color1)
        axes[env_id].axvspan(20000, 40000, alpha=0.1, color=color2)
        axes[env_id].axvspan(40000, 60000, alpha=0.15, color=color3)
        axes[env_id].axvspan(60000, 70000, alpha=0.1, color=color1)
        for i in range(len(settings)):
            act_perfs_mean = np.load(FILE_PATH + 'avg_perfs_mean_' + settings[i] + '.npy')
            act_perfs_std = np.load(FILE_PATH + 'avg_perfs_std_' + settings[i] + '.npy')
            time_stamps = np.load(FILE_PATH + 'time_stamps_' + settings[i] + '.npy')
            axes[env_id].plot(time_stamps, act_perfs_mean[env_id, :], linewidth=linewidths[i], color=line_colors[i], label=labels[i])
            axes[env_id].set_xlabel('Trials')
            axes[env_id].set_ylabel('Performance')
            # axes[env_id].set_title('Task{:d} Performance'.format(env_id+1))
            axes[env_id].set_xlim([0.0, 71000])
            axes[env_id].set_ylim([0.0, 1.01])
            axes[env_id].set_xticks([0, 20000, 40000, 60000, 70000])
            axes[env_id].set_yticks([0.2*i for i in range(6)])
            axes[env_id].set_xticklabels([0, 20, 40, 60, 70])
            axes[env_id].set_yticklabels([round(0.2*i, 1) for i in range(6)])
    axes[-1].legend(bbox_to_anchor = (1.0, 0.65))
    plt.tight_layout()
    # plt.show()
    plt.savefig('./files/' + 'threetasksperformance-baselines.pdf')
    plt.close()

# PFC+MD VS baselines: cases
if 0:
    FILE_PATH = './files/example_cases/two_tasks/'
    settings = ['PFCMD', 'EWC', 'SI', 'PFC']
    line_colors = ['tab:red', 'violet', 'tab:green', 'tab:blue']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC+SI', 'PFC']
    linewidths = [2, 1.5, 1.5, 1.5]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for env_id in range(2): # 2 tasks
        color1, color2= 'tab:red', 'tab:blue'
        axes[env_id].axvspan(    0, 20000, alpha=0.08, color=color1)
        axes[env_id].axvspan(20000, 40000, alpha=0.08, color=color2)
        axes[env_id].axvspan(40000, 50000, alpha=0.08, color=color1)
        for i in range(len(settings)):
            setting = settings[i]
            config = np.load(FILE_PATH + '346_config_' + setting + '.npy', allow_pickle=True).item()
            log = np.load(FILE_PATH + '346_log_' + setting + '.npy', allow_pickle=True).item()
            time_stamps = np.array(log.stamps)
            act_perfs = np.array(log.act_perfs)
            axes[env_id].plot(time_stamps, act_perfs[env_id, :], linewidth=linewidths[i], color=line_colors[i], label=labels[i])
            axes[env_id].set_xlabel('Trials')
            axes[env_id].set_ylabel('Performance')
            axes[env_id].set_title('Task{:d}: '.format(env_id+1) + config.task_seq[env_id][len('yang19.'):-len('-v0')])
            axes[env_id].set_xlim([0.0, 51000])
            axes[env_id].set_ylim([0.0, 1.01])
            axes[env_id].set_xticks([0, 20000, 40000, 50000])
            axes[env_id].set_yticks([0.2*j for j in range(6)])
            axes[env_id].set_xticklabels([0, 20, 40, 50])
            axes[env_id].set_yticklabels([round(0.2*j, 1) for j in range(6)])
    axes[-1].legend(bbox_to_anchor = (1.0, 0.65))
    plt.tight_layout(w_pad=3.0)
    plt.show()
    # plt.savefig('./files/' + 'twotasksperformance-cases.pdf')
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

# Record activity
if 0:
    FILE_PATH = './files/trajectory/PFC/'
    log = np.load(FILE_PATH + 'log.npy', allow_pickle=True).item()
    config = np.load(FILE_PATH + 'config.npy', allow_pickle=True).item()
    dataset = np.load(FILE_PATH + 'dataset.npy', allow_pickle=True).item()
    net = torch.load(FILE_PATH + 'net.pt')
    crit = nn.MSELoss()

    # turn on test mode
    net.eval()
    if hasattr(config, 'MDeffect'):
        if config.MDeffect:
            net.rnn.md.learn = False
    # testing
    with torch.no_grad():
        for task_id in [0, 1]:
            inputs, labels = dataset(task_id=task_id)
            outputs, rnn_activity = net(inputs, task_id=task_id)
            loss = crit(outputs, labels)
            np.save('./files/'+f'PFC_activity_{task_id}.npy', rnn_activity)
            if hasattr(config, 'MDeffect'):
                if config.MDeffect:
                    np.save('./files/'+f'MD_activity_{task_id}.npy', net.rnn.md.md_output_t)
                    np.save('./files/'+f'PFC_ctx_activity_{task_id}.npy', net.rnn.PFC_ctx_acts)
            print(loss)

# PFC trajectory
if 0:
    mode = 'PFC'
    if mode == 'PFCMD':
        FILE_PATH = './files/trajectory/PFCMD/'
    elif mode == 'PFC':
        FILE_PATH = './files/trajectory/PFC/'

    # fit PCA
    PFC_activity = []
    for task_id in [0, 1]:
        PFC_activity.append(np.mean(np.load(FILE_PATH+f'PFC_activity_{task_id}.npy'), axis=1))
    PFC_activity_concat = np.concatenate(PFC_activity, axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(PFC_activity_concat)
    print('explained variance ratio:', pca.explained_variance_ratio_)
    print('explained variance', pca.explained_variance_)
    PFC_activity_reduced = []
    for activity in PFC_activity:
        PFC_activity_reduced.append(pca.transform(activity))

    # make plots
    plt.figure(figsize=(4, 4))
    points = np.array([6, 13, 15]) if mode == 'PFCMD' else np.array([5, 10, 14]) # plot a few arrow of these points
    X = PFC_activity_reduced[0][points, 0]
    Y = PFC_activity_reduced[0][points, 1]
    U = PFC_activity_reduced[0][points+1, 0]-PFC_activity_reduced[0][points, 0]
    V = PFC_activity_reduced[0][points+1, 1]-PFC_activity_reduced[0][points, 1]
    norm = np.sqrt(np.square(U) + np.square(V)) # make arrow size constant
    plt.quiver(X, Y, U/norm, V/norm,
               scale_units='xy', angles='xy', scale=1.5, width=0.01, color='tab:orange')
    plt.plot(PFC_activity_reduced[0][:, 0], PFC_activity_reduced[0][:, 1],
             c='tab:orange', marker='', linewidth=3.0, alpha=1.0, label='Task1')
    plt.plot(PFC_activity_reduced[0][-1, 0], PFC_activity_reduced[0][-1, 1],
             c='tab:orange', marker='*', markersize=15, alpha=1.0)
    points = np.array([0, 20, 30]) if mode == 'PFCMD' else np.array([2, 7, 24])
    X = PFC_activity_reduced[1][points, 0]
    Y = PFC_activity_reduced[1][points, 1]
    U = PFC_activity_reduced[1][points+1, 0]-PFC_activity_reduced[1][points, 0]
    V = PFC_activity_reduced[1][points+1, 1]-PFC_activity_reduced[1][points, 1]
    norm = np.sqrt(np.square(U) + np.square(V))
    plt.quiver(X, Y, U/norm, V/norm,
               scale_units='xy', angles='xy', scale=1.5, width=0.01, color='tab:green')
    plt.plot(PFC_activity_reduced[1][:, 0], PFC_activity_reduced[1][:, 1],
             c='tab:green', marker='', linewidth=3.0, alpha=1.0, label='Task2')
    plt.plot(PFC_activity_reduced[1][-1, 0], PFC_activity_reduced[1][-1, 1],
             c='tab:green', marker='*', markersize=15, alpha=1.0)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PC1', labelpad=15)
    plt.ylabel('PC2', labelpad=15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.0))
    # plt.title('PFC activity of a trial')
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'trajectory_' + mode + '.pdf')
    # plt.close()

# MD trajectory
if 0:
    # fit PCA
    MD_activity = []
    for task_id in [0, 1]:
        MD_activity.append(np.load('./files/'+f'MD_activity_{task_id}.npy'))
    MD_activity_concat = np.concatenate(MD_activity, axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(MD_activity_concat)
    print('explained variance ratio:', pca.explained_variance_ratio_)
    print('explained variance', pca.explained_variance_)
    MD_activity_reduced = []
    for activity in MD_activity:
        MD_activity_reduced.append(pca.transform(activity))

    # make plots
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    plt.figure(figsize=(6, 6))
    plt.scatter(MD_activity_reduced[0][:, 0], MD_activity_reduced[0][:, 1],
                c='tab:red', marker='o', s=100, alpha=1.0, label='Task1')
    plt.scatter(MD_activity_reduced[1][:, 0], MD_activity_reduced[1][:, 1],
                c='tab:blue', marker='o', s=100, alpha=1.0, label='Task2')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PC1', fontdict=label_font)
    plt.ylabel('PC2', fontdict=label_font)
    plt.legend(bbox_to_anchor = (1.2, 0.6), prop=legend_font)
    plt.title('MD activity in two tasks', fontdict=title_font)
    plt.show()

# PFC-ctx trajectory
if 0:
    # fit PCA
    PFC_ctx_activity = []
    for task_id in [0, 1]:
        PFC_ctx_activity.append(np.load('./files/'+f'PFC_ctx_activity_{task_id}.npy'))
    PFC_ctx_activity_concat = np.concatenate(PFC_ctx_activity, axis=0)
    
    pca = PCA(n_components=2)
    pca.fit(PFC_ctx_activity_concat)
    print('explained variance ratio:', pca.explained_variance_ratio_)
    print('explained variance', pca.explained_variance_)
    PFC_ctx_activity_reduced = []
    for activity in PFC_ctx_activity:
        PFC_ctx_activity_reduced.append(pca.transform(activity))

    # make plots
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    plt.figure(figsize=(6, 6))
    plt.scatter(PFC_ctx_activity_reduced[0][:, 0], PFC_ctx_activity_reduced[0][:, 1], c='tab:red', marker='o', alpha=1.0, label='Task1')
    plt.scatter(PFC_ctx_activity_reduced[1][:, 0], PFC_ctx_activity_reduced[1][:, 1], c='tab:blue', marker='o', alpha=1.0, label='Task2')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PC1', fontdict=label_font)
    plt.ylabel('PC2', fontdict=label_font)
    plt.legend(bbox_to_anchor = (1.2, 0.6), prop=legend_font)
    plt.title('PFC_ctx activity in two tasks', fontdict=title_font)
    plt.show()

# Connections weights
if 0:
    FILE_PATH = './files/trajectory/PFCMD/'
    # FILE_PATH = './files/randomortho_init/cases/'
    log = np.load(FILE_PATH + 'log.npy', allow_pickle=True).item()
    config = np.load(FILE_PATH + 'config.npy', allow_pickle=True).item()
    dataset = np.load(FILE_PATH + 'dataset.npy', allow_pickle=True).item()
    net = torch.load(FILE_PATH + 'net.pt')

    # winput2PFC-ctx
    fig, axes = plt.subplots(figsize=(5, 4))
    ax = axes
    sns.heatmap(net.rnn.input2PFCctx.weight.data, cmap='Reds', ax=ax, vmin=0, vmax=0.05)
    ax.set_xticks([0, config.input_size])
    ax.set_xticklabels([1, config.input_size], rotation=0)
    ax.set_yticks([0, config.hidden_ctx_size])
    ax.set_yticklabels([1, config.hidden_ctx_size], rotation=0)
    ax.set_xlabel('Input Units')
    ax.set_ylabel('PFC-ctx Neurons')
    # ax.set_title('Input to PFC-ctx weights')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Connection Weight', labelpad=15)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'weights_winput2PFC-ctx.pdf')
    # plt.close()

    # wPFC-ctx2MD
    fig, axes = plt.subplots(figsize=(5, 4))
    ax = axes
    ax = sns.heatmap(net.rnn.md.wPFC2MD, cmap='Reds', ax=ax, vmin=0, vmax=2)
    ax.set_xticks([0, config.hidden_ctx_size-1])
    ax.set_xticklabels([1, config.hidden_ctx_size], rotation=0)
    ax.set_yticklabels([i+1 for i in range(config.md_size)], rotation=0)
    ax.set_xlabel('PFC-ctx Neurons')
    ax.set_ylabel('MD Neurons')
    # ax.set_title('wPFC2MD')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cbar.set_ticklabels([0, 0.5, 1.0, 1.5, 2.0])
    cbar.set_label('Connection Weight', labelpad=15)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'weights_wPFC-ctx2MD.pdf')
    # plt.close()

    # wMD2PFC
    fig, axes = plt.subplots(figsize=(5, 4))
    ax = axes
    sns.heatmap(net.rnn.md.wMD2PFC, cmap='Blues_r', ax=ax, vmin=-5, vmax=0)
    ax.set_xticklabels([i+1 for i in range(config.md_size)], rotation=0)
    ax.set_yticks([0, config.hidden_size-1])
    ax.set_yticklabels([1, config.hidden_size], rotation=0)
    ax.set_xlabel('MD Neurons')
    ax.set_ylabel('PFC Neurons')
    # ax.set_title('MD to PFC weights')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, -1, -2, -3, -4, -5])
    cbar.set_ticklabels([0, -1, -2, -3, -4, -5])
    cbar.set_label('Connection Weight', labelpad=15)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'weights_wMD2PFC.pdf')
    # plt.close()

    # wPFCtoPFC
    fig, axes = plt.subplots(figsize=(6, 6))
    ax = axes
    cax = ax.matshow(net.rnn.h2h.weight.data, cmap='magma')
    ax.set_xticks([0, config.hidden_size-1])
    ax.set_xticklabels([1, config.hidden_size], rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0, config.hidden_size-1])
    ax.set_yticklabels([1, config.hidden_size], rotation=0)
    ax.set_xlabel('PFC Neurons')
    ax.set_ylabel('PFC Neurons')
    ax.spines['left'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(0)
    # ax.set_title('PFC to PFC weights')
    cbar = fig.colorbar(cax, **{'fraction':0.046, 'pad':0.04})
    cbar.set_label('Connection Weight', labelpad=15)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./files/' + 'weights_wPFC2PFC.pdf')
    plt.close()

    # wPFCtoPFC with masked diagonal elements
    fig, axes = plt.subplots(figsize=(6, 6))
    ax = axes
    data = net.rnn.h2h.weight.data.numpy()
    np.fill_diagonal(data, 0)
    cax = ax.matshow(data, cmap='magma', vmax=0.02, vmin=-0.02)
    ax.set_xticks([0, config.hidden_size-1])
    ax.set_xticklabels([1, config.hidden_size], rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0, config.hidden_size-1])
    ax.set_yticklabels([1, config.hidden_size], rotation=0)
    ax.set_xlabel('PFC Neurons')
    ax.set_ylabel('PFC Neurons')
    ax.spines['left'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(0)
    # ax.set_title('PFC to PFC weights')
    cbar = fig.colorbar(cax, **{'fraction':0.046, 'pad':0.04})
    cbar.set_ticks([-0.02, 0, 0.02])
    cbar.set_label('Connection Weight', labelpad=15)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'weights_wPFC2PFC_maskdiagonal.pdf')
    # plt.close()

    # winputtoPFC
    fig, axes = plt.subplots(figsize=(5, 6))
    ax = axes
    sns.heatmap(net.rnn.input2h.weight.data, cmap='magma', ax=ax)
    ax.set_xticks([0, config.input_size])
    ax.set_xticklabels([1, config.input_size], rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0, config.hidden_size-1])
    ax.set_yticklabels([1, config.hidden_size], rotation=0)
    ax.set_xlabel('Input Units')
    ax.set_ylabel('PFC Neurons')
    # ax.set_title('Input to PFC weights')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Connection Weight', labelpad=15)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'weights_winput2PFC.pdf')
    # plt.close()

    # wPFCtooutput
    fig, axes = plt.subplots(figsize=(7, 4))
    ax = axes
    sns.heatmap(net.fc.weight.data, cmap='magma', ax=ax)
    ax.set_xticks([0, config.hidden_size-1])
    ax.set_xticklabels([1, config.hidden_size], rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks([0, config.output_size])
    ax.set_yticklabels([1, config.output_size], rotation=0)
    ax.set_xlabel('PFC Neurons')
    ax.set_ylabel('Output Units')
    # ax.set_title('Input to PFC weights')
    cbar = ax.collections[0].colorbar
    cbar.set_label('Connection Weight', labelpad=15)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./files/' + 'weights_wPFC2output.pdf')
    # plt.close()

# the evolution of PFC-ctx to MD weight
if 0:
    FILE_PATH = './files/weight_evolution/two_tasks/'
    config = np.load(FILE_PATH + 'config.npy', allow_pickle=True).item()
    for trial_num in [0, 19999, 39999, 49999]: # two tasks
    # for trial_num in [0, 19999, 39999, 59999, 69999]: # three tasks
        data = np.load(FILE_PATH + f'wPFC-ctx2MD_trial{trial_num}.npy')
        fig, axes = plt.subplots(figsize=(5, 4))
        ax = axes
        ax = sns.heatmap(data, cmap='Reds', ax=ax, vmin=0, vmax=2)
        ax.set_xticks([0, config.hidden_ctx_size-1])
        ax.set_xticklabels([1, config.hidden_ctx_size], rotation=0)
        ax.set_yticklabels([i+1 for i in range(config.md_size)], rotation=0)
        ax.set_xlabel('PFC-ctx Neurons')
        ax.set_ylabel('MD Neurons')
        # ax.set_title('wPFC2MD')
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
        cbar.set_ticklabels([0, 0.5, 1.0, 1.5, 2.0])
        cbar.set_label('Connection Weight', labelpad=15)
        cbar.outline.set_linewidth(1.2)
        cbar.ax.tick_params(labelsize=12, width=1.2)
        plt.tight_layout()
        plt.show()
        # plt.savefig('./files/' + f'wPFC-ctx2MD_trial{trial_num}.pdf')
        # plt.close()

# how PFCMD_Rikhye doesn't work?
if 0:
    FILE_PATH = './files/PFCMD_Rikhye/'
    for trial_num in [j*4000-1 for j in range(1, 13)]: # two tasks
        data = np.load(FILE_PATH + f'wPFC2MD_trial{trial_num}.npy')
        fig, axes = plt.subplots(figsize=(5, 4))
        ax = axes
        ax = sns.heatmap(data, cmap='Reds', ax=ax, vmin=0, vmax=2)
        ax.set_xticks([0, 600-1])
        ax.set_xticklabels([1, 600], rotation=0)
        ax.set_yticklabels([j+1 for j in range(2)], rotation=0)
        ax.set_xlabel('PFC Neurons')
        ax.set_ylabel('MD Neurons')
        # ax.set_title('wPFC2MD')
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
        cbar.set_ticklabels([0, 0.5, 1.0, 1.5, 2.0])
        cbar.set_label('Connection Weight', labelpad=15)
        cbar.outline.set_linewidth(1.2)
        cbar.ax.tick_params(labelsize=12, width=1.2)
        plt.tight_layout()
        plt.show()
        # plt.savefig('./files/' + f'wPFC2MD_trial{trial_num}.pdf')
        # plt.close()

# Task similarity analysis
if 0:
    if 0:
        # Retrieve Yang 2019 results
        FILE_PATH = './files/similarity/'
        tick_names = np.load(FILE_PATH + 'tick_names.npy')
        norm_task_variance = np.load(FILE_PATH + 'norm_task_variance.npy')
        task_ids = []
        for i in range(len(tick_names)):
            if tick_names[i] not in ['dlydm1', 'dlydm2', 'ctxdlydm1', 'ctxdlydm2', 'multidlydm']: # we don't use delayed DM family
                task_ids.append(i)
        tick_names = tick_names[task_ids]
        tick_names_dict = dict()
        for id, tick_name in enumerate(tick_names):
            tick_names_dict[str(id)] = tick_name
        tick_names_dict_reversed = {v:k for k, v in tick_names_dict.items()}
        
        norm_task_variance = norm_task_variance[task_ids, :]
        similarity_matrix = norm_task_variance @ norm_task_variance.T
        # normalized by max
        # max_similarity_matrix = np.amax(similarity_matrix, axis=0)
        # norm_similarity_matrix = 0.5 * (similarity_matrix/max_similarity_matrix[np.newaxis, :] +
        #                                 similarity_matrix/max_similarity_matrix[:, np.newaxis])
        # normalized by diagonal elements
        # diag_similarity_matrix = np.diag(similarity_matrix)
        # norm_similarity_matrix = 0.5 * (similarity_matrix/diag_similarity_matrix[np.newaxis, :] +
        #                                 similarity_matrix/diag_similarity_matrix[:, np.newaxis])
        # normalized Euclidean norm
        Euclid_norm = np.sqrt(np.diag(similarity_matrix))
        norm_similarity_matrix = similarity_matrix / np.outer(Euclid_norm, Euclid_norm)

        # heatmap norm_task_variance
        fig, axes = plt.subplots(figsize=(6, 4))
        ax = axes
        ax = sns.heatmap(norm_task_variance,
                        vmin=0, vmax=1, cmap='hot', ax=ax,
                        cbar_kws={'fraction':0.046, 'pad':0.04})
        plt.xticks([])
        plt.yticks([i+0.5 for i in range(len(tick_names))], tick_names, 
                rotation=0, va='center', font='arial', fontsize=12)
        # plt.title('Units')
        plt.xlabel('Units')
        ax.tick_params('both', length=0)
        cbar = ax.collections[0].colorbar
        cbar.outline.set_linewidth(1.2)
        cbar.set_label('Normalized Task Variance', labelpad=15)
        cbar.ax.tick_params(labelsize=12, width=1.2)
        # cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # cbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.tight_layout()
        plt.show()
        # plt.savefig('./files/' + 'norm_task_variance.pdf')
        # plt.close()

        # heatmap norm_similarity_matrix
        fig, axes = plt.subplots(figsize=(5, 4))
        ax = axes
        mask = np.zeros_like(norm_similarity_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[np.diag_indices_from(mask)] = False
        ax = sns.heatmap(norm_similarity_matrix,
                        vmin=0, vmax=1, mask=mask,
                        cmap='OrRd', ax=ax, square=True,
                        cbar_kws={'fraction':0.046, 'pad':0.04})
        ax.xaxis.tick_bottom()
        plt.xticks([i+0.5 for i in range(len(tick_names))], tick_names, 
                rotation='vertical', ha='center', font='arial', fontsize=12)
        plt.yticks([i+0.5 for i in range(len(tick_names))], tick_names, 
                rotation=0, va='center', font='arial', fontsize=12)
        ax.tick_params('both', length=0)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Normalized Task Similarity', labelpad=15)
        cbar.outline.set_linewidth(1.2)
        cbar.ax.tick_params(labelsize=12, width=1.2)
        plt.tight_layout()
        plt.show()
        # plt.savefig('./files/' + 'norm_similarity_matrix.pdf')
        # plt.close()

        # select task seqs with high similarity
        sim_task_seqs = []
        similarity_threshold = 0.5
        for sim_task_seq_id in np.argwhere(norm_similarity_matrix > similarity_threshold):
            if sim_task_seq_id[0] == sim_task_seq_id[1]: # exclude two identical tasks
                continue
            sim_task_seqs.append(
                ('yang19.' + tick_names_dict[f'{sim_task_seq_id[0]}'] + '-v0',
                'yang19.' + tick_names_dict[f'{sim_task_seq_id[1]}'] + '-v0')
            )
    '''
    sim_task_seqs based on similarity_matrix (threshold = 0.6):
    [('yang19.go-v0', 'yang19.rtgo-v0'), ('yang19.go-v0', 'yang19.dlygo-v0'),
    ('yang19.go-v0', 'yang19.anti-v0'), ('yang19.go-v0', 'yang19.dlyanti-v0'),
    ('yang19.go-v0', 'yang19.dm1-v0'), ('yang19.go-v0', 'yang19.dm2-v0'),
    ('yang19.rtgo-v0', 'yang19.go-v0'), ('yang19.rtgo-v0', 'yang19.rtanti-v0'),
    ('yang19.dlygo-v0', 'yang19.go-v0'), ('yang19.dlygo-v0', 'yang19.anti-v0'),
    ('yang19.dlygo-v0', 'yang19.dlyanti-v0'), ('yang19.anti-v0', 'yang19.go-v0'),
    ('yang19.anti-v0', 'yang19.dlygo-v0'), ('yang19.anti-v0', 'yang19.rtanti-v0'),
    ('yang19.anti-v0', 'yang19.dlyanti-v0'), ('yang19.rtanti-v0', 'yang19.rtgo-v0'),
    ('yang19.rtanti-v0', 'yang19.anti-v0'), ('yang19.rtanti-v0', 'yang19.dlyanti-v0'),
    ('yang19.dlyanti-v0', 'yang19.go-v0'), ('yang19.dlyanti-v0', 'yang19.dlygo-v0'),
    ('yang19.dlyanti-v0', 'yang19.anti-v0'), ('yang19.dlyanti-v0', 'yang19.rtanti-v0'),
    ('yang19.dm1-v0', 'yang19.go-v0'), ('yang19.dm1-v0', 'yang19.dm2-v0'),
    ('yang19.dm1-v0', 'yang19.ctxdm1-v0'), ('yang19.dm1-v0', 'yang19.ctxdm2-v0'),
    ('yang19.dm1-v0', 'yang19.multidm-v0'), ('yang19.dm2-v0', 'yang19.go-v0'),
    ('yang19.dm2-v0', 'yang19.dm1-v0'), ('yang19.dm2-v0', 'yang19.ctxdm1-v0'),
    ('yang19.dm2-v0', 'yang19.ctxdm2-v0'), ('yang19.dm2-v0', 'yang19.multidm-v0'),
    ('yang19.ctxdm1-v0', 'yang19.dm1-v0'), ('yang19.ctxdm1-v0', 'yang19.dm2-v0'),
    ('yang19.ctxdm1-v0', 'yang19.ctxdm2-v0'), ('yang19.ctxdm1-v0', 'yang19.multidm-v0'),
    ('yang19.ctxdm2-v0', 'yang19.dm1-v0'), ('yang19.ctxdm2-v0', 'yang19.dm2-v0'),
    ('yang19.ctxdm2-v0', 'yang19.ctxdm1-v0'), ('yang19.ctxdm2-v0', 'yang19.multidm-v0'),
    ('yang19.multidm-v0', 'yang19.dm1-v0'), ('yang19.multidm-v0', 'yang19.dm2-v0'),
    ('yang19.multidm-v0', 'yang19.ctxdm1-v0'), ('yang19.multidm-v0', 'yang19.ctxdm2-v0'),
    ('yang19.dms-v0', 'yang19.dnms-v0'), ('yang19.dms-v0', 'yang19.dmc-v0'),
    ('yang19.dms-v0', 'yang19.dnmc-v0'), ('yang19.dnms-v0', 'yang19.dms-v0'),
    ('yang19.dnms-v0', 'yang19.dmc-v0'), ('yang19.dnms-v0', 'yang19.dnmc-v0'),
    ('yang19.dmc-v0', 'yang19.dms-v0'), ('yang19.dmc-v0', 'yang19.dnms-v0'),
    ('yang19.dmc-v0', 'yang19.dnmc-v0'), ('yang19.dnmc-v0', 'yang19.dms-v0'),
    ('yang19.dnmc-v0', 'yang19.dnms-v0'), ('yang19.dnmc-v0', 'yang19.dmc-v0')]
    
    Based on the meaning of task, removed some task seqs in sim_task_seqs:
    1. One task is in Go task family, and the other is in Anti task family. e.g. ('yang19.go-v0', 'yang19.anti-v0').
    2. One task is a Match task, and the other is a non-match task. e.g. ('yang19.dms-v0', 'yang19.dnms-v0').
    
    sim_task_seqs = \
    [('yang19.go-v0', 'yang19.rtgo-v0'), ('yang19.go-v0', 'yang19.dlygo-v0'),
    ('yang19.go-v0', 'yang19.dm1-v0'), ('yang19.go-v0', 'yang19.dm2-v0'),
    ('yang19.rtgo-v0', 'yang19.go-v0'), ('yang19.dlygo-v0', 'yang19.go-v0'),
    ('yang19.anti-v0', 'yang19.rtanti-v0'), ('yang19.anti-v0', 'yang19.dlyanti-v0'),
    ('yang19.rtanti-v0', 'yang19.anti-v0'), ('yang19.rtanti-v0', 'yang19.dlyanti-v0'),
    ('yang19.dlyanti-v0', 'yang19.anti-v0'), ('yang19.dlyanti-v0', 'yang19.rtanti-v0'),
    ('yang19.dm1-v0', 'yang19.go-v0'), ('yang19.dm1-v0', 'yang19.dm2-v0'),
    ('yang19.dm1-v0', 'yang19.ctxdm1-v0'), ('yang19.dm1-v0', 'yang19.ctxdm2-v0'),
    ('yang19.dm1-v0', 'yang19.multidm-v0'), ('yang19.dm2-v0', 'yang19.go-v0'),
    ('yang19.dm2-v0', 'yang19.dm1-v0'), ('yang19.dm2-v0', 'yang19.ctxdm1-v0'),
    ('yang19.dm2-v0', 'yang19.ctxdm2-v0'), ('yang19.dm2-v0', 'yang19.multidm-v0'),
    ('yang19.ctxdm1-v0', 'yang19.dm1-v0'), ('yang19.ctxdm1-v0', 'yang19.dm2-v0'),
    ('yang19.ctxdm1-v0', 'yang19.ctxdm2-v0'), ('yang19.ctxdm1-v0', 'yang19.multidm-v0'),
    ('yang19.ctxdm2-v0', 'yang19.dm1-v0'), ('yang19.ctxdm2-v0', 'yang19.dm2-v0'),
    ('yang19.ctxdm2-v0', 'yang19.ctxdm1-v0'), ('yang19.ctxdm2-v0', 'yang19.multidm-v0'),
    ('yang19.multidm-v0', 'yang19.dm1-v0'), ('yang19.multidm-v0', 'yang19.dm2-v0'),
    ('yang19.multidm-v0', 'yang19.ctxdm1-v0'), ('yang19.multidm-v0', 'yang19.ctxdm2-v0'),
    ('yang19.dms-v0', 'yang19.dmc-v0'), ('yang19.dnms-v0', 'yang19.dnmc-v0'),
    ('yang19.dmc-v0', 'yang19.dms-v0'), ('yang19.dnmc-v0', 'yang19.dnms-v0')]

    sim_task_seqs based on similarity_matrix (threshold = 0.5):
    [('yang19.go-v0', 'yang19.rtgo-v0'), ('yang19.go-v0', 'yang19.dlygo-v0'),
     ('yang19.go-v0', 'yang19.anti-v0'), ('yang19.go-v0', 'yang19.dlyanti-v0'),
     ('yang19.go-v0', 'yang19.dm1-v0'), ('yang19.go-v0', 'yang19.dm2-v0'),
     ('yang19.go-v0', 'yang19.ctxdm2-v0'), ('yang19.go-v0', 'yang19.multidm-v0'),
     ('yang19.go-v0', 'yang19.dnmc-v0'), ('yang19.rtgo-v0', 'yang19.go-v0'),
     ('yang19.rtgo-v0', 'yang19.dlygo-v0'), ('yang19.rtgo-v0', 'yang19.anti-v0'),
     ('yang19.rtgo-v0', 'yang19.rtanti-v0'), ('yang19.rtgo-v0', 'yang19.dm1-v0'),
     ('yang19.rtgo-v0', 'yang19.dm2-v0'), ('yang19.rtgo-v0', 'yang19.ctxdm1-v0'),
     ('yang19.rtgo-v0', 'yang19.ctxdm2-v0'), ('yang19.rtgo-v0', 'yang19.multidm-v0'),
     ('yang19.rtgo-v0', 'yang19.dnmc-v0'), ('yang19.dlygo-v0', 'yang19.go-v0'),
     ('yang19.dlygo-v0', 'yang19.rtgo-v0'), ('yang19.dlygo-v0', 'yang19.anti-v0'),
     ('yang19.dlygo-v0', 'yang19.dlyanti-v0'), ('yang19.anti-v0', 'yang19.go-v0'),
     ('yang19.anti-v0', 'yang19.rtgo-v0'), ('yang19.anti-v0', 'yang19.dlygo-v0'),
     ('yang19.anti-v0', 'yang19.rtanti-v0'), ('yang19.anti-v0', 'yang19.dlyanti-v0'),
     ('yang19.rtanti-v0', 'yang19.rtgo-v0'), ('yang19.rtanti-v0', 'yang19.anti-v0'),
     ('yang19.rtanti-v0', 'yang19.dlyanti-v0'), ('yang19.dlyanti-v0', 'yang19.go-v0'),
     ('yang19.dlyanti-v0', 'yang19.dlygo-v0'), ('yang19.dlyanti-v0', 'yang19.anti-v0'),
     ('yang19.dlyanti-v0', 'yang19.rtanti-v0'), ('yang19.dm1-v0', 'yang19.go-v0'),
     ('yang19.dm1-v0', 'yang19.rtgo-v0'), ('yang19.dm1-v0', 'yang19.dm2-v0'),
     ('yang19.dm1-v0', 'yang19.ctxdm1-v0'), ('yang19.dm1-v0', 'yang19.ctxdm2-v0'),
     ('yang19.dm1-v0', 'yang19.multidm-v0'), ('yang19.dm2-v0', 'yang19.go-v0'),
     ('yang19.dm2-v0', 'yang19.rtgo-v0'), ('yang19.dm2-v0', 'yang19.dm1-v0'),
     ('yang19.dm2-v0', 'yang19.ctxdm1-v0'), ('yang19.dm2-v0', 'yang19.ctxdm2-v0'),
     ('yang19.dm2-v0', 'yang19.multidm-v0'), ('yang19.ctxdm1-v0', 'yang19.rtgo-v0'),
     ('yang19.ctxdm1-v0', 'yang19.dm1-v0'), ('yang19.ctxdm1-v0', 'yang19.dm2-v0'),
     ('yang19.ctxdm1-v0', 'yang19.ctxdm2-v0'), ('yang19.ctxdm1-v0', 'yang19.multidm-v0'),
     ('yang19.ctxdm2-v0', 'yang19.go-v0'), ('yang19.ctxdm2-v0', 'yang19.rtgo-v0'),
     ('yang19.ctxdm2-v0', 'yang19.dm1-v0'), ('yang19.ctxdm2-v0', 'yang19.dm2-v0'),
     ('yang19.ctxdm2-v0', 'yang19.ctxdm1-v0'), ('yang19.ctxdm2-v0', 'yang19.multidm-v0'),
     ('yang19.multidm-v0', 'yang19.go-v0'), ('yang19.multidm-v0', 'yang19.rtgo-v0'),
     ('yang19.multidm-v0', 'yang19.dm1-v0'), ('yang19.multidm-v0', 'yang19.dm2-v0'),
     ('yang19.multidm-v0', 'yang19.ctxdm1-v0'), ('yang19.multidm-v0', 'yang19.ctxdm2-v0'),
     ('yang19.dms-v0', 'yang19.dnms-v0'), ('yang19.dms-v0', 'yang19.dmc-v0'),
     ('yang19.dms-v0', 'yang19.dnmc-v0'), ('yang19.dnms-v0', 'yang19.dms-v0'),
     ('yang19.dnms-v0', 'yang19.dmc-v0'), ('yang19.dnms-v0', 'yang19.dnmc-v0'),
     ('yang19.dmc-v0', 'yang19.dms-v0'), ('yang19.dmc-v0', 'yang19.dnms-v0'),
     ('yang19.dmc-v0', 'yang19.dnmc-v0'), ('yang19.dnmc-v0', 'yang19.go-v0'),
     ('yang19.dnmc-v0', 'yang19.rtgo-v0'), ('yang19.dnmc-v0', 'yang19.dms-v0'),
     ('yang19.dnmc-v0', 'yang19.dnms-v0'), ('yang19.dnmc-v0', 'yang19.dmc-v0')]
    '''

    sim_task_seqs = \
    [('yang19.go-v0', 'yang19.rtgo-v0'), ('yang19.go-v0', 'yang19.dlygo-v0'),
     ('yang19.go-v0', 'yang19.anti-v0'), ('yang19.go-v0', 'yang19.dlyanti-v0'),
     ('yang19.go-v0', 'yang19.dm1-v0'), ('yang19.go-v0', 'yang19.dm2-v0'),
     ('yang19.go-v0', 'yang19.ctxdm2-v0'), ('yang19.go-v0', 'yang19.multidm-v0'),
     ('yang19.go-v0', 'yang19.dnmc-v0'), ('yang19.rtgo-v0', 'yang19.go-v0'),
     ('yang19.rtgo-v0', 'yang19.dlygo-v0'), ('yang19.rtgo-v0', 'yang19.anti-v0'),
     ('yang19.rtgo-v0', 'yang19.rtanti-v0'), ('yang19.rtgo-v0', 'yang19.dm1-v0'),
     ('yang19.rtgo-v0', 'yang19.dm2-v0'), ('yang19.rtgo-v0', 'yang19.ctxdm1-v0'),
     ('yang19.rtgo-v0', 'yang19.ctxdm2-v0'), ('yang19.rtgo-v0', 'yang19.multidm-v0'),
     ('yang19.rtgo-v0', 'yang19.dnmc-v0'), ('yang19.dlygo-v0', 'yang19.go-v0'),
     ('yang19.dlygo-v0', 'yang19.rtgo-v0'), ('yang19.dlygo-v0', 'yang19.anti-v0'),
     ('yang19.dlygo-v0', 'yang19.dlyanti-v0'), ('yang19.anti-v0', 'yang19.go-v0'),
     ('yang19.anti-v0', 'yang19.rtgo-v0'), ('yang19.anti-v0', 'yang19.dlygo-v0'),
     ('yang19.anti-v0', 'yang19.rtanti-v0'), ('yang19.anti-v0', 'yang19.dlyanti-v0'),
     ('yang19.rtanti-v0', 'yang19.rtgo-v0'), ('yang19.rtanti-v0', 'yang19.anti-v0'),
     ('yang19.rtanti-v0', 'yang19.dlyanti-v0'), ('yang19.dlyanti-v0', 'yang19.go-v0'),
     ('yang19.dlyanti-v0', 'yang19.dlygo-v0'), ('yang19.dlyanti-v0', 'yang19.anti-v0'),
     ('yang19.dlyanti-v0', 'yang19.rtanti-v0'), ('yang19.dm1-v0', 'yang19.go-v0'),
     ('yang19.dm1-v0', 'yang19.rtgo-v0'), ('yang19.dm1-v0', 'yang19.dm2-v0'),
     ('yang19.dm1-v0', 'yang19.ctxdm1-v0'), ('yang19.dm1-v0', 'yang19.ctxdm2-v0'),
     ('yang19.dm1-v0', 'yang19.multidm-v0'), ('yang19.dm2-v0', 'yang19.go-v0'),
     ('yang19.dm2-v0', 'yang19.rtgo-v0'), ('yang19.dm2-v0', 'yang19.dm1-v0'),
     ('yang19.dm2-v0', 'yang19.ctxdm1-v0'), ('yang19.dm2-v0', 'yang19.ctxdm2-v0'),
     ('yang19.dm2-v0', 'yang19.multidm-v0'), ('yang19.ctxdm1-v0', 'yang19.rtgo-v0'),
     ('yang19.ctxdm1-v0', 'yang19.dm1-v0'), ('yang19.ctxdm1-v0', 'yang19.dm2-v0'),
     ('yang19.ctxdm1-v0', 'yang19.ctxdm2-v0'), ('yang19.ctxdm1-v0', 'yang19.multidm-v0'),
     ('yang19.ctxdm2-v0', 'yang19.go-v0'), ('yang19.ctxdm2-v0', 'yang19.rtgo-v0'),
     ('yang19.ctxdm2-v0', 'yang19.dm1-v0'), ('yang19.ctxdm2-v0', 'yang19.dm2-v0'),
     ('yang19.ctxdm2-v0', 'yang19.ctxdm1-v0'), ('yang19.ctxdm2-v0', 'yang19.multidm-v0'),
     ('yang19.multidm-v0', 'yang19.go-v0'), ('yang19.multidm-v0', 'yang19.rtgo-v0'),
     ('yang19.multidm-v0', 'yang19.dm1-v0'), ('yang19.multidm-v0', 'yang19.dm2-v0'),
     ('yang19.multidm-v0', 'yang19.ctxdm1-v0'), ('yang19.multidm-v0', 'yang19.ctxdm2-v0'),
     ('yang19.dms-v0', 'yang19.dnms-v0'), ('yang19.dms-v0', 'yang19.dmc-v0'),
     ('yang19.dms-v0', 'yang19.dnmc-v0'), ('yang19.dnms-v0', 'yang19.dms-v0'),
     ('yang19.dnms-v0', 'yang19.dmc-v0'), ('yang19.dnms-v0', 'yang19.dnmc-v0'),
     ('yang19.dmc-v0', 'yang19.dms-v0'), ('yang19.dmc-v0', 'yang19.dnms-v0'),
     ('yang19.dmc-v0', 'yang19.dnmc-v0'), ('yang19.dnmc-v0', 'yang19.go-v0'),
     ('yang19.dnmc-v0', 'yang19.rtgo-v0'), ('yang19.dnmc-v0', 'yang19.dms-v0'),
     ('yang19.dnmc-v0', 'yang19.dnms-v0'), ('yang19.dnmc-v0', 'yang19.dmc-v0')]

    # match the task seqs with high similarity with those in the scale up test
    scaleup_task_seqs = np.load('./files/similarity/scaleup_task_seqs.npy').tolist()
    take_seq_ids = []
    for sim_task_seq in sim_task_seqs:
        take_seq_ids += get_task_seq_id(task_seqs=scaleup_task_seqs, task_seq=sim_task_seq)
    take_seq_ids.sort()
    # update the sim_task_seqs
    sim_task_seqs = []
    for take_seq_id in take_seq_ids:
        sim_task_seqs.append(scaleup_task_seqs[take_seq_id])
    print(take_seq_ids)
    print(sim_task_seqs)
    
    # compute performance
    # 1. non-similar task seqs + original PFCMD
    # FILE_PATH = './files/scaleup_twotasks_5/PFCMD/'
    # SAVE_FILE_NAME = 'nonsimilar_PFCMD'
    # settings = ['PFCMD']
    # ITER = list(range(420))
    # ITER = list(set(ITER) - set(take_seq_ids))
    # LEN = len(ITER)
    # 2. non-similar task seqs + reduced PFCMD
    # FILE_PATH = './files/similarity/scaleup_twotasks_3/'
    # SAVE_FILE_NAME = 'nonsimilar_reducedPFCMD'
    # settings = ['PFCMD']
    # ITER = list(range(420))
    # ITER = list(set(ITER) - set(take_seq_ids))
    # LEN = len(ITER)
    # 3. similar task seqs + original PFCMD
    # FILE_PATH = './files/scaleup_twotasks_5/PFCMD/'
    # SAVE_FILE_NAME = 'similar_PFCMD'
    # settings = ['PFCMD']
    # ITER = list(take_seq_ids)
    # LEN = len(ITER)
    # 4. similar task seqs + reduced PFCMD
    # FILE_PATH = './files/similarity/scaleup_twotasks_3/'
    # SAVE_FILE_NAME = 'similar_reducedPFCMD'
    # settings = ['PFCMD']
    # ITER = list(take_seq_ids)
    # LEN = len(ITER)
    # 5. similar task seqs + PFC
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'similar_PFC'
    # settings = ['PFC']
    # ITER = take_seq_ids
    # LEN = len(ITER)
    # 6. non-similar task seqs + PFC
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'nonsimilar_PFC'
    # settings = ['PFC']
    # ITER = list(range(420))
    # ITER = list(set(ITER) - set(take_seq_ids))
    # LEN = len(ITER)
    # 7. similar task seqs + PFCEWC
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'similar_PFCEWC'
    # settings = ['EWC']
    # ITER = take_seq_ids
    # LEN = len(ITER)
    # 8. non-similar task seqs + PFCEWC
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'nonsimilar_PFCEWC'
    # settings = ['EWC']
    # ITER = list(range(420))
    # ITER = list(set(ITER) - set(take_seq_ids))
    # LEN = len(ITER)
    # 9. similar task seqs + PFCSI
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'similar_PFCSI'
    # settings = ['SI']
    # ITER = take_seq_ids
    # LEN = len(ITER)
    # 10. non-similar task seqs + PFCSI
    FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    SAVE_FILE_NAME = 'nonsimilar_PFCSI'
    settings = ['SI']
    ITER = list(range(420))
    ITER = list(set(ITER) - set(take_seq_ids))
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

    # plot performance
    fig, axes = plt.subplots(figsize=(4, 4))
    ax = axes
    line_colors = ['tab:orange', 'tab:green']
    labels = ['Task1', 'Task2']
    linewidth = 2
    ax.axvspan(    0, 20000, alpha=0.1, color=line_colors[0])
    ax.axvspan(20000, 40000, alpha=0.1, color=line_colors[1])
    ax.axvspan(40000, 50000, alpha=0.1, color=line_colors[0])
    for env_id in range(2): # 2 tasks
        plt.plot(time_stamps, act_perfs_mean[env_id, :],
                 linewidth=linewidth, color=line_colors[env_id], label=labels[env_id])
        plt.fill_between(time_stamps, 
                         act_perfs_mean[env_id, :]-act_perfs_std[env_id, :],
                         act_perfs_mean[env_id, :]+act_perfs_std[env_id, :],
                         alpha=0.2, color=line_colors[env_id])
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    # plt.title('Task Performance')
    plt.xlim([0.0, 51000])
    plt.ylim([0.0, 1.01])
    plt.xticks(ticks=[0, 20000, 40000, 50000], labels=[0, 20, 40, 50])
    plt.yticks([0.2*i for i in range(6)])
    plt.legend(loc='lower right', bbox_to_anchor=(0.99, 0.05))
    plt.tight_layout()
    # plt.show()
    plt.savefig('./files/' + SAVE_FILE_NAME + '.pdf')
    plt.close()

# forward transfer, continual learning VS task similarity
if 0:
    FILE_PATH = './files/similarity/'
    tick_names = np.load(FILE_PATH + 'tick_names.npy')
    tick_names_dict = np.load(FILE_PATH + 'tick_names_dict.npy', allow_pickle=True).item()
    tick_names_dict_reversed = np.load(FILE_PATH + 'tick_names_dict_reversed.npy', allow_pickle=True).item()
    norm_similarity_matrix = np.load(FILE_PATH + 'norm_similarity_matrix.npy')

    # forward transfer, continual learning and task similarity of each task pair
    # 1. original PFCMD
    # FILE_PATH = './files/scaleup_twotasks_5/PFCMD/'
    # SAVE_FILE_NAME = 'PFCMD'
    # setting = 'PFCMD'
    # 2. reduced PFCMD
    # FILE_PATH = './files/similarity/scaleup_twotasks_3/'
    # SAVE_FILE_NAME = 'reducedPFCMD'
    # setting = 'PFCMD'
    # 3. PFC
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'PFC'
    # setting = 'PFC'
    # 4. PFCEWC
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'PFCEWC'
    # setting = 'EWC'
    # 5. PFCSI
    # FILE_PATH = './files/scaleup_twotasks_5/baselines/'
    # SAVE_FILE_NAME = 'PFCSI'
    # setting = 'SI'

    forward_transfer_perf = np.zeros(shape=(420))
    continual_learning_perf = np.zeros(shape=(420))
    task_similarity = np.zeros(shape=(420))
    for i in range(420):
        config = np.load(FILE_PATH + str(i) + '_config_' + setting + '.npy', allow_pickle=True).item()
        log = np.load(FILE_PATH + str(i) + '_log_' + setting + '.npy', allow_pickle=True).item()
        forward_transfer_perf[i] = np.array(log.act_perfs)[1, 39]
        continual_learning_perf[i] = np.array(log.act_perfs)[0, 79]
        x = int(tick_names_dict_reversed[config.task_seq[0][len('yang19.'):-len('-v0')]])
        y = int(tick_names_dict_reversed[config.task_seq[1][len('yang19.'):-len('-v0')]])
        task_similarity[i] = norm_similarity_matrix[x, y]
    
    # 6. binary control: similar -> overlapping; dissimilar -> disjoint
    # SAVE_FILE_NAME = 'binarycontrol'
    # setting = 'PFCMD'
    # scaleup_task_seqs = np.load('./files/similarity/scaleup_task_seqs.npy').tolist()
    # threshold = 0.5
    # forward_transfer_perf = np.zeros(shape=(420))
    # continual_learning_perf = np.zeros(shape=(420))
    # task_similarity = np.zeros(shape=(420))
    # for i in range(420):
    #     # fetch task similarity
    #     task_seq = scaleup_task_seqs[i]
    #     x = int(tick_names_dict_reversed[task_seq[0][len('yang19.'):-len('-v0')]])
    #     y = int(tick_names_dict_reversed[task_seq[1][len('yang19.'):-len('-v0')]])
    #     task_similarity[i] = norm_similarity_matrix[x, y]
    #     # binary control
    #     if task_similarity[i] > threshold:
    #         FILE_PATH = './files/similarity/scaleup_twotasks_3/'
    #     else:
    #         FILE_PATH = './files/scaleup_twotasks_5/PFCMD/'
    #     config = np.load(FILE_PATH + str(i) + '_config_' + setting + '.npy', allow_pickle=True).item()
    #     log = np.load(FILE_PATH + str(i) + '_log_' + setting + '.npy', allow_pickle=True).item()
    #     forward_transfer_perf[i] = np.array(log.act_perfs)[1, 39]
    #     continual_learning_perf[i] = np.array(log.act_perfs)[0, 79]
    
    # different ways to plot the data
    # 1. split data into a few intervals
    if 0:
        task_similarity_data = np.array([0.1*i for i in range(3, 11)])
        L = len(task_similarity_data)
        forward_transfer_perf_data = np.zeros(shape=(L))
        forward_transfer_perf_std_data = np.zeros(shape=(L))
        continual_learning_perf_data = np.zeros(shape=(L))
        continual_learning_perf_std_data = np.zeros(shape=(L))
        for x in range(L):
            similarity = task_similarity_data[x]
            ids = []
            for y in range(420):
                if task_similarity[y] >= similarity-0.05 and task_similarity[y] < similarity+0.05:
                    ids.append(y)
            forward_transfer_perf_data[x] = np.mean(forward_transfer_perf[ids])
            forward_transfer_perf_std_data[x] = np.std(forward_transfer_perf[ids])
            continual_learning_perf_data[x] = np.mean(continual_learning_perf[ids])
            continual_learning_perf_std_data[x] = np.std(continual_learning_perf[ids])
        
        # plot as lines
        if 0:
            fig, axes = plt.subplots(figsize=(4, 4))
            ax = axes
            line_colors = ['deeppink', 'deepskyblue']
            labels = ['CL', 'FT']
            linewidth = 2
            # fill is better than errorbar
            # plt.errorbar(task_similarity_data, forward_transfer_perf_data, yerr=forward_transfer_perf_std_data,
            #              linewidth=linewidth, color=line_colors[0], label=labels[0])
            # plt.errorbar(task_similarity_data, continual_learning_perf_data, yerr=continual_learning_perf_std_data,
            #              linewidth=linewidth, color=line_colors[1], label=labels[1])
            plt.plot(task_similarity_data, continual_learning_perf_data,
                    linewidth=linewidth, color=line_colors[0], label=labels[0])
            plt.fill_between(task_similarity_data, 
                            continual_learning_perf_data - continual_learning_perf_std_data,
                            continual_learning_perf_data + continual_learning_perf_std_data,
                            alpha=0.1, color=line_colors[0])
            plt.plot(task_similarity_data, forward_transfer_perf_data,
                    linewidth=linewidth, color=line_colors[1], label=labels[1])
            plt.fill_between(task_similarity_data, 
                            forward_transfer_perf_data - forward_transfer_perf_std_data,
                            forward_transfer_perf_data + forward_transfer_perf_std_data,
                            alpha=0.1, color=line_colors[1])
            
            plt.xlabel('Task Similarity')
            plt.ylabel('Performance')
            plt.xlim([0.3, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xticks(ticks=[round(0.1*i, 1) for i in range(3, 11)], labels=[round(0.1*i, 1) for i in range(3, 11)])
            plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.legend(loc='lower right', bbox_to_anchor=(1.05, 0.1))
            plt.tight_layout()
            plt.show()
            # plt.savefig('./files/' + 'FTCL_VS_similarity_' + SAVE_FILE_NAME + '.pdf')
            # plt.close()
        
        # plot as bars
        if 1:
            fig, axes = plt.subplots(figsize=(6, 4))
            ax = axes
            bar_width = 0.35
            line_colors = ['orchid', 'deepskyblue']
            labels = ['CL', 'FT']
            error_kw = {'ecolor':'lightgray', 'capsize':0.0}
            index_CL = np.arange(len(task_similarity_data))
            index_FT = index_CL + bar_width
            plt.bar(x=index_CL, height=continual_learning_perf_data,
                    color=line_colors[0], width=bar_width, label=labels[0],
                    yerr=continual_learning_perf_std_data,
                    error_kw=error_kw)
            plt.bar(x=index_FT, height=forward_transfer_perf_data,
                    color=line_colors[1], width=bar_width, label=labels[1],
                    yerr=forward_transfer_perf_std_data,
                    error_kw=error_kw)
            plt.xticks(index_CL + bar_width/2, [round(0.1*i, 1) for i in range(3, 11)])
            plt.xlabel('Task Similarity')
            plt.ylabel('Performance')
            plt.ylim([0.0, 1.01])
            plt.legend(bbox_to_anchor=(0.98, 0.6))
            plt.tight_layout()
            # plt.show()
            plt.savefig('./files/' + 'FTCL_VS_similarity_' + SAVE_FILE_NAME + '.pdf')
            plt.close()
    
    # 2. scatter plot and linear regression
    if 1:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        line_colors = ['orchid', 'deepskyblue']
        labels = ['CL', 'FT']
        # subplot 1
        axes[0].scatter(task_similarity, continual_learning_perf, c=line_colors[0], s=10, alpha=0.5)
        axes[0].set_xlabel('Task Similarity')
        axes[0].set_ylabel('CL Performance')
        axes[0].set_xlim([0.3, 1.0])
        axes[0].set_ylim([0.0, 1.01])
        axes[0].set_xticks(ticks=[round(0.1*i, 1) for i in range(3, 11)])
        axes[0].set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        reg = LinearRegression()
        reg.fit(task_similarity.reshape(-1, 1), continual_learning_perf.reshape(-1, 1))
        X_pred = np.linspace(0.3, 1.0, 100)
        y_pred = reg.predict(X_pred.reshape(-1, 1)).squeeze()
        axes[0].plot(X_pred, y_pred, linewidth=2.5, color=line_colors[0])
        # subplot 2
        axes[1].scatter(task_similarity, forward_transfer_perf, c=line_colors[1], s=10, alpha=0.5)
        axes[1].set_xlim([0.3, 1.0])
        axes[1].set_ylim([0.0, 1.01])
        axes[1].set_xticks(ticks=[round(0.1*i, 1) for i in range(3, 11)])
        axes[1].set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[1].set_xlabel('Task Similarity')
        axes[1].set_ylabel('FT Performance')
        reg = LinearRegression()
        reg.fit(task_similarity.reshape(-1, 1), forward_transfer_perf.reshape(-1, 1))
        X_pred = np.linspace(0.3, 1.0, 100)
        y_pred = reg.predict(X_pred.reshape(-1, 1)).squeeze()
        axes[1].plot(X_pred, y_pred, linewidth=2.5, color=line_colors[1])
        # show
        plt.tight_layout(w_pad=3.0)
        plt.show()
        # plt.savefig('./files/' + 'FTCL_VS_similarity_scatterplot_' + SAVE_FILE_NAME + '.pdf')
        # plt.close()


# decoding analysis
# train and score linear decoder
if 0:
    dir_names = os.listdir('./files/MDisnecessary/')
    for dir_name in tqdm(dir_names):
        if dir_name in ['backup', 'settings.txt']:
            continue
        FILE_PATH = './files/MDisnecessary/' + dir_name + '/'
        log = np.load(FILE_PATH + 'log.npy', allow_pickle=True).item()
        config = np.load(FILE_PATH + 'config.npy', allow_pickle=True).item()
        dataset = np.load(FILE_PATH + 'dataset.npy', allow_pickle=True).item()
        net = torch.load(FILE_PATH + 'net.pt')
        
        num_test = 10
        num_trials = 500
        num_train_trials = 400

        # turn on test mode
        net.eval()
        if hasattr(config, 'MDeffect'):
            if config.MDeffect:
                net.rnn.md.learn = False
        # prepare train & test data for decoder
        with torch.no_grad():
            PFCctx_scores = []
            MD_scores = []
            for _ in range(num_test):
                PFCctx_train, MD_train, context_train = [], [], []
                PFCctx_test, MD_test, context_test = [], [], []
                for i in range(num_trials):
                    # randomly choose a task
                    task_id = random.choice(range(config.num_task))
                    # record PFCctx and MD activities
                    inputs, labels = dataset(task_id=task_id)
                    outputs, rnn_activity = net(inputs, task_id=task_id)
                    # context labels
                    context_label = np.zeros(shape=(inputs.shape[0], config.num_task))
                    context_label[:, task_id] = 1
                    if i < num_train_trials:
                        PFCctx_train.append(net.rnn.PFC_ctx_acts.copy())
                        MD_train.append(net.rnn.md.md_output_t.copy())
                        context_train.append(context_label.copy())
                    else:
                        PFCctx_test.append(net.rnn.PFC_ctx_acts.copy())
                        MD_test.append(net.rnn.md.md_output_t.copy())
                        context_test.append(context_label.copy())
                PFCctx_train = np.concatenate(PFCctx_train, axis=0)
                MD_train = np.concatenate(MD_train, axis=0)
                context_train = np.concatenate(context_train, axis=0)
                PFCctx_test = np.concatenate(PFCctx_test, axis=0)
                MD_test = np.concatenate(MD_test, axis=0)
                context_test = np.concatenate(context_test, axis=0)
            
                # train & score linear decoder
                PFCctx_reg = LinearRegression()
                PFCctx_reg.fit(PFCctx_train, context_train)
                PFCctx_score = PFCctx_reg.score(PFCctx_test, context_test)
                PFCctx_scores.append(PFCctx_score)
                MD_reg = LinearRegression()
                MD_reg.fit(MD_train, context_train)
                MD_score = MD_reg.score(MD_test, context_test)
                MD_scores.append(MD_score)
            
            # save mean & std of score
            np.save(FILE_PATH + 'PFCctx_score_mean', np.mean(PFCctx_scores))
            np.save(FILE_PATH + 'PFCctx_score_std', np.std(PFCctx_scores))
            np.save(FILE_PATH + 'MD_score_mean', np.mean(MD_scores))
            np.save(FILE_PATH + 'MD_score_std', np.std(MD_scores))

# plot scores VS noise std
if 0:
    noise_stds = ['0dot0001', '0dot001', '0dot01', '0dot02', '0dot03', '0dot04', '0dot05', '0dot06', '0dot07', '0dot08']
    sub_active_prob = '0dot40'
    noise_std_data = []
    PFCctx_scores_mean = []
    PFCctx_scores_std = []
    MD_scores_mean = []
    MD_scores_std = []
    for noise_std in noise_stds:
        FILE_PATH = './files/MDisnecessary/noisestd' + noise_std + 'prob' + sub_active_prob +'/'
        config = np.load(FILE_PATH + 'config.npy', allow_pickle=True).item()
        noise_std_data.append(config.hidden_ctx_noise)
        PFCctx_scores_mean.append(np.load(FILE_PATH + 'PFCctx_score_mean.npy').item())
        PFCctx_scores_std.append(np.load(FILE_PATH + 'PFCctx_score_std.npy').item())
        MD_scores_mean.append(np.load(FILE_PATH + 'MD_score_mean.npy').item())
        MD_scores_std.append(np.load(FILE_PATH + 'MD_score_std.npy').item())
    PFCctx_scores_mean = np.array(PFCctx_scores_mean)
    PFCctx_scores_std = np.array(PFCctx_scores_std)
    MD_scores_mean = np.array(MD_scores_mean)
    MD_scores_std = np.array(MD_scores_std)

    plt.figure(figsize=(4, 4))
    plt.semilogx(noise_std_data, MD_scores_mean, '-s', color='tab:blue', linewidth=3, markersize=9, label='MD')
    plt.semilogx(noise_std_data, PFCctx_scores_mean, '-v', color='#f47e76', linewidth=3, markersize=9, label='PFC-ctx')
    plt.fill_between(noise_std_data,
                     MD_scores_mean - MD_scores_std,
                     np.clip(MD_scores_mean + MD_scores_std, 0, 1),
                     alpha=0.2, color='tab:blue')
    plt.fill_between(noise_std_data,
                     PFCctx_scores_mean - PFCctx_scores_std,
                     np.clip(PFCctx_scores_mean + PFCctx_scores_std, 0, 1),
                     alpha=0.2, color='#f47e76')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0.05))
    plt.xlabel('Noise STD')
    plt.ylabel('Model Score') # R-squared is the Coefficient of Determination
    plt.ylim([0.8, 1.01])
    plt.yticks([0.1*i for i in range(8, 11)])
    plt.tight_layout()
    # plt.show()
    plt.savefig('./files/' + 'decoding_vs_noisestd.pdf')
    plt.close()

# plot scores VS activated probability
if 0:
    noise_std = '0dot01'
    sub_active_probs = ['0dot04', '0dot05', '0dot06', '0dot07', '0dot08', '0dot09', '0dot10', '0dot40', '0dot70', '1dot00']
    sub_active_prob_data = []
    PFCctx_scores_mean = []
    PFCctx_scores_std = []
    MD_scores_mean = []
    MD_scores_std = []
    for sub_active_prob in sub_active_probs:
        FILE_PATH = './files/MDisnecessary/noisestd' + noise_std + 'prob' + sub_active_prob +'/'
        config = np.load(FILE_PATH + 'config.npy', allow_pickle=True).item()
        sub_active_prob_data.append(config.sub_active_prob)
        PFCctx_scores_mean.append(np.load(FILE_PATH + 'PFCctx_score_mean.npy').item())
        PFCctx_scores_std.append(np.load(FILE_PATH + 'PFCctx_score_std.npy').item())
        MD_scores_mean.append(np.load(FILE_PATH + 'MD_score_mean.npy').item())
        MD_scores_std.append(np.load(FILE_PATH + 'MD_score_std.npy').item())
    PFCctx_scores_mean = np.array(PFCctx_scores_mean)
    PFCctx_scores_std = np.array(PFCctx_scores_std)
    MD_scores_mean = np.array(MD_scores_mean)
    MD_scores_std = np.array(MD_scores_std)

    plt.figure(figsize=(4, 4))
    plt.plot(sub_active_prob_data, MD_scores_mean, '-s', color='tab:blue', linewidth=3, markersize=9, label='MD')
    plt.plot(sub_active_prob_data, PFCctx_scores_mean, '-v', color='#f47e76', linewidth=3, markersize=9, label='PFC-ctx')
    plt.fill_between(sub_active_prob_data,
                     MD_scores_mean - MD_scores_std,
                     np.clip(MD_scores_mean + MD_scores_std, 0, 1),
                     alpha=0.2, color='tab:blue')
    plt.fill_between(sub_active_prob_data,
                     PFCctx_scores_mean - PFCctx_scores_std,
                     np.clip(PFCctx_scores_mean + PFCctx_scores_std, 0, 1),
                     alpha=0.2, color='#f47e76')
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.05))
    plt.xlabel('Activated Probability', labelpad=14) 
    plt.ylabel('Model Score') # R-squared is the Coefficient of Determination
    plt.xlim([0, 1.08])
    plt.ylim([0.6, 1.01])
    plt.xticks([0.2*i for i in range(6)])
    plt.yticks([0.1*i for i in range(6, 11)])
    plt.tight_layout()
    # plt.show()
    plt.savefig('./files/' + 'decoding_vs_activatedprob.pdf')
    plt.close()

# plot inputs and outputs of neurogym tasks
if 0:
    tasks = ngym.get_collection('yang19')
    env_kwargs = {'dt': 100}
    RNGSEED = 5
    random.seed(RNGSEED)
    np.random.seed(RNGSEED)
    torch.manual_seed(RNGSEED)
    for task in tasks:
        task_name = task[len('yang19.'):-len('-v0')]
        env = gym.make(task, **env_kwargs)
        env.new_trial()
        ob, gt = env.ob, env.gt # ob:(trial_len, input_size); gt:(trial_len)
        trial_len = gt.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig.suptitle('Task: ' + task_name)
        sns.heatmap(ob.T, cmap='Reds', ax=axes[0])
        axes[0].set_xticks([0.5, trial_len-0.5])
        axes[0].set_xticklabels([0, trial_len], rotation=0)
        axes[0].set_yticks([0.5, 32.5])
        axes[0].set_yticklabels([0, 33], rotation=0)
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Input Units')
        cbar = axes[0].collections[0].colorbar
        cbar.outline.set_linewidth(1.2)
        cbar.ax.tick_params(labelsize=12, width=1.2)
        axes[1].plot(gt.T)
        axes[1].set_xticks([0, trial_len-1])
        axes[1].set_xticklabels([1, trial_len], rotation=0)
        axes[1].set_xlim([0, trial_len-1])
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Ground Truth')
        plt.tight_layout(w_pad=3.0)
        plt.show()
        # plt.savefig('./files/' + 'obgt_' + task_name + '.pdf')
        # plt.close()