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
import imageio
from pygifsicle import optimize

# RNN activities
def plot_rnn_activity(rnn_activity):
    font = {'family':'Times New Roman','weight':'normal', 'size':20}
    plt.figure()
    plt.plot(rnn_activity[-1, 0, :].cpu().detach().numpy())
    plt.title('PFC activities', fontdict=font)
    plt.show()

# MD related variables
def plot_MD_variables(net, config):
    font = {'family':'Times New Roman','weight':'normal', 'size':20}
    # Presynaptic traces
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 2, 1)
    plt.plot(net.rnn.md.md_preTraces[-1, :])
    plt.axhline(y=net.rnn.md.md_preTrace_thresholds[-1], color='r', linestyle='-')
    plt.title('Pretrace', fontdict=font)
    # Binary presynaptic traces
    sub_size = config.sub_size
    plt.subplot(2, 2, 2)
    plt.plot(net.rnn.md.md_preTraces_binary[-1, :])
    plt.axhline(y=net.rnn.md.md_preTrace_binary_thresholds[-1], color='r', linestyle='-')
    plt.title( 'Pretrace_binary\n' +
                f'L: {sum(net.rnn.md.md_preTraces_binary[-1, :sub_size]) / sub_size}; ' +
                f'R: {sum(net.rnn.md.md_preTraces_binary[-1, sub_size:]) / sub_size}; ' +
                f'ALL: {sum(net.rnn.md.md_preTraces_binary[-1, :]) / len(net.rnn.md.md_preTraces_binary[-1, :])}',
                fontdict=font)
    # MD activities
    plt.subplot(2, 2, 3)
    plt.plot(net.rnn.md.md_output_t[-1, :])
    plt.title('MD activities', fontdict=font)
    # Heatmap wPFC2MD
    # ax = plt.subplot(2, 2, 4)
    # ax = sns.heatmap(net.rnn.md.wPFC2MD, cmap='Reds')
    # ax.set_xticks([0, config.hidden_ctx_size-1])
    # ax.set_xticklabels([1, config.hidden_ctx_size], rotation=0)
    # ax.set_yticklabels([i for i in range(config.md_size)], rotation=0)
    # ax.set_xlabel('PFC neuron index', fontdict=font)
    # ax.set_ylabel('MD neuron index', fontdict=font)
    # ax.set_title('wPFC2MD', fontdict=font)
    # cbar = ax.collections[0].colorbar
    # cbar.set_label('connection weight', fontdict=font)
    ## Heatmap wMD2PFC
    ax = plt.subplot(2, 2, 4)
    ax = sns.heatmap(net.rnn.md.wMD2PFC, cmap='Blues_r')
    ax.set_xticklabels([i for i in range(config.md_size)], rotation=0)
    ax.set_yticks([0, config.hidden_size-1])
    ax.set_yticklabels([1, config.hidden_size], rotation=0)
    ax.set_xlabel('MD neuron index', fontdict=font)
    ax.set_ylabel('PFC neuron index', fontdict=font)
    ax.set_title('wMD2PFC', fontdict=font)
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight', fontdict=font)
    ## Heatmap wMD2PFCMult
    # font = {'family':'Times New Roman','weight':'normal', 'size':20}
    # ax = plt.subplot(2, 3, 6)
    # ax = sns.heatmap(net.rnn.md.wMD2PFCMult, cmap='Reds')
    # ax.set_xticklabels([i for i in range(config.md_size)], rotation=0)
    # ax.set_yticks([0, config.hidden_size-1])
    # ax.set_yticklabels([1, config.hidden_size], rotation=0)
    # ax.set_xlabel('MD neuron index', fontdict=font)
    # ax.set_ylabel('PFC neuron index', fontdict=font)
    # ax.set_title('wMD2PFCMult', fontdict=font)
    # cbar = ax.collections[0].colorbar
    # cbar.set_label('connection weight', fontdict=font)
    plt.tight_layout()
    plt.show()

# loss curve
def plot_loss(log):
    font = {'family':'Times New Roman','weight':'normal', 'size':25}
    plt.figure()
    plt.plot(np.array(log.losses))
    plt.xlabel('Trials', fontdict=font)
    plt.ylabel('Training MSE loss', fontdict=font)
    plt.tight_layout()
    # plt.savefig('./animation/'+'CEloss.png')
    plt.show()

# performance curve (fixation performance and action performance)
def plot_fullperf(config, log):
    label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':25}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
    for env_id in range(config.num_task):
        plt.figure()
        plt.plot(log.stamps, log.fix_perfs[env_id], label='fix')
        plt.plot(log.stamps, log.act_perfs[env_id], label='act')
        plt.fill_between(x=[config.switch_points[0], config.switch_points[1]], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.fill_between(x=[config.switch_points[1], config.switch_points[2]], y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
        plt.fill_between(x=[config.switch_points[2], config.total_trials    ], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.legend(bbox_to_anchor = (1.15, 0.7), prop=legend_font)
        plt.xlabel('Trials', fontdict=label_font)
        plt.ylabel('Performance', fontdict=label_font)
        plt.title('Task{:d}: '.format(env_id+1)+config.taskpair[env_id], fontdict=title_font)
        # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
        plt.xlim([0.0, None])
        plt.ylim([0.0, 1.01])
        plt.yticks([0.1*i for i in range(11)])
        plt.tight_layout()
        # plt.savefig('./animation/'+'performance.png')
        plt.show()

# performance curve
def plot_perf(config, log):
    label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':25}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
    for env_id in range(config.num_task):
        plt.figure()
        plt.plot(log.stamps, log.act_perfs[env_id], color='red', label='$ MD+ $')
        plt.fill_between(x=[config.switch_points[0], config.switch_points[1]], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.fill_between(x=[config.switch_points[1], config.switch_points[2]], y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
        plt.fill_between(x=[config.switch_points[2], config.total_trials    ], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.legend(bbox_to_anchor = (1.25, 0.7), prop=legend_font)
        plt.xlabel('Trials', fontdict=label_font)
        plt.ylabel('Performance', fontdict=label_font)
        plt.title('Task{:d}: '.format(env_id+1)+config.taskpair[env_id], fontdict=title_font)
        # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
        plt.xlim([0.0, None])
        plt.ylim([0.0, 1.01])
        plt.yticks([0.1*i for i in range(11)])
        plt.tight_layout()
        # plt.savefig('./animation/'+'performance.png')
        plt.show()