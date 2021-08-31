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