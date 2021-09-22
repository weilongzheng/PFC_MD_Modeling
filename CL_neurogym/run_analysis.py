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
import seaborn as sns


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


from sklearn.decomposition import PCA
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
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    plt.figure(figsize=(5, 4))
    plt.quiver(PFC_activity_reduced[0][:-1, 0],
               PFC_activity_reduced[0][:-1, 1],
               PFC_activity_reduced[0][1:, 0]-PFC_activity_reduced[0][:-1, 0],
               PFC_activity_reduced[0][1:, 1]-PFC_activity_reduced[0][:-1, 1],
               scale_units='xy', angles='xy', scale=1.2, width=0.01, color='tab:red')
    plt.plot(PFC_activity_reduced[0][:, 0], PFC_activity_reduced[0][:, 1],
             c='tab:red', marker='', linewidth=2.5, alpha=1.0, label='Task1')
    plt.quiver(PFC_activity_reduced[1][:-1, 0],
               PFC_activity_reduced[1][:-1, 1],
               PFC_activity_reduced[1][1:, 0]-PFC_activity_reduced[1][:-1, 0],
               PFC_activity_reduced[1][1:, 1]-PFC_activity_reduced[1][:-1, 1],
               scale_units='xy', angles='xy', scale=1.2, width=0.01, color='tab:blue')
    plt.plot(PFC_activity_reduced[1][:, 0], PFC_activity_reduced[1][:, 1],
             c='tab:blue', marker='', linewidth=2.5, alpha=1.0, label='Task2')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('PC1', fontdict=label_font)
    plt.ylabel('PC2', fontdict=label_font)
    plt.legend(bbox_to_anchor = (1.3, 0.6), prop=legend_font)
    plt.title('PFC activity of a trial', fontdict=title_font)
    plt.tight_layout()
    plt.show()
    # plt.savefig(FILE_PATH + 'trajectory.pdf')
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

# PFC trajectory
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

# MD related weights
if 0:
    FILE_PATH = './files/trajectory/PFCMD/'
    log = np.load(FILE_PATH + 'log.npy', allow_pickle=True).item()
    config = np.load(FILE_PATH + 'config.npy', allow_pickle=True).item()
    dataset = np.load(FILE_PATH + 'dataset.npy', allow_pickle=True).item()
    net = torch.load(FILE_PATH + 'net.pt')

    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    # winput2PFC-ctx
    fig, axes = plt.subplots(figsize=(6, 4))
    ax = axes
    sns.heatmap(net.rnn.input2PFCctx.weight.data, cmap='Reds', ax=ax, vmin=0, vmax=0.05)
    ax.set_xticks([0, config.input_size])
    ax.set_xticklabels([1, config.input_size], rotation=0)
    ax.set_yticks([0, config.hidden_ctx_size])
    ax.set_yticklabels([1, config.hidden_ctx_size], rotation=0)
    ax.set_xlabel('Input index', fontdict=label_font)
    ax.set_ylabel('PFC-ctx index', fontdict=label_font)
    ax.set_title('Input to PFC-ctx weights', fontdict=title_font)
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight', fontdict=label_font)
    plt.show()
    # plt.savefig(FILE_PATH + 'weights_winput2PFC-ctx.pdf')
    # plt.close()
    # wMD2PFC
    fig, axes = plt.subplots(figsize=(6, 4))
    ax = axes
    sns.heatmap(net.rnn.md.wMD2PFC, cmap='Blues_r', ax=ax, vmin=-5, vmax=0)
    ax.set_xticklabels([i+1 for i in range(config.md_size)], rotation=0)
    ax.set_yticks([0, config.hidden_size-1])
    ax.set_yticklabels([1, config.hidden_size], rotation=0)
    ax.set_xlabel('MD index', fontdict=label_font)
    ax.set_ylabel('PFC index', fontdict=label_font)
    ax.set_title('MD to PFC weights', fontdict=title_font)
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight', fontdict=label_font)
    plt.show()
    # plt.savefig(FILE_PATH + 'weights_wMD2PFC.pdf')
    # plt.close()