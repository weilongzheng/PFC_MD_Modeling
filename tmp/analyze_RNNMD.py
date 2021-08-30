import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import imageio
from pygifsicle import optimize


# log = np.load('./files/'+'log_withMD.npy', allow_pickle=True).item()


# PFC outputs within a cycle
if False:
    PFCouts_all = log['PFCouts_all']
    print(len(PFCouts_all))
    font = {'family':'Times New Roman','weight':'normal', 'size':25}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    for idx_cycle in [-1]: # choose the cycle to visualize
        PFCouts_cycle = PFCouts_all[idx_cycle].squeeze()
        for i in range(PFCouts_cycle.shape[0]):
            # meanPFCouts_cycle = np.mean(PFCouts_cycle[i, :])
            # plt.axhline(y=meanPFCouts_cycle, color='r', linestyle='-')
            plt.plot(PFCouts_cycle[i, :])
            plt.title(f'PFC outputs Cycle-{idx_cycle}' + ' Step-'+str(i+1), fontdict=font)
            plt.xlabel('PFC neuron index', fontdict=label_font)
            plt.ylabel('PFC activities', fontdict=label_font)
            plt.xlim([-5, 261])
            plt.ylim([0.0, 1.0])
            plt.savefig('./animation/'+f'PFCoutputs_index_{i}.png')
            plt.close() # do not show figs in line
        images = []
        for i in range(PFCouts_cycle.shape[0]):
            filename = './animation/'+f'PFCoutputs_index_{i}.png'
            images.append(imageio.imread(filename))
        gif_path = './animation/'+f'PFCoutputs_evolution_cycle{idx_cycle}.gif'
        imageio.mimsave(gif_path, images, duration=0.2)
        optimize(gif_path)

# Presynaptic traces within a cycle
if False:
    MDpreTraces_all = log['MDpreTraces_all']
    MDpreTrace_threshold_all = log['MDpreTrace_threshold_all']
    font = {'family':'Times New Roman','weight':'normal', 'size':25}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    for idx_cycle in [0, 1, 149, 150, 151, 152, 153, 199, 200, 250, 399, 400]: # choose the cycle to visualize
        MDpreTraces_cycle = MDpreTraces_all[idx_cycle, :, :]
        MDpreTrace_threshold_cycle = MDpreTrace_threshold_all[idx_cycle, :, :]
        for i in range(MDpreTraces_cycle.shape[0]):
            plt.plot(MDpreTraces_cycle[i, :])
            plt.axhline(y=MDpreTrace_threshold_cycle[i, :], color='r', linestyle='-')
            plt.title(f'Pre traces Cycle-{idx_cycle}' + ' Step-'+str(i+1), fontdict=font)
            plt.xlabel('PFC neuron index', fontdict=label_font)
            plt.ylabel('Presynaptic traces', fontdict=label_font)
            plt.xlim([-5, 261])
            plt.ylim([0.0, 0.1])
            plt.savefig('./animation/'+f'MDpreTraces_index_{i}.png')
            plt.close() # do not show figs in line
        images = []
        for i in range(MDpreTraces_cycle.shape[0]):
            filename = './animation/'+f'MDpreTraces_index_{i}.png'
            images.append(imageio.imread(filename))
        gif_path = './animation/'+f'MDpreTraces_evolution_cycle{idx_cycle}.gif'
        imageio.mimsave(gif_path, images, duration=0.2)
        optimize(gif_path)

if False:
    if config['MDeffect']:
        # Heatmap wPFC2MD
        font = {'family':'Times New Roman','weight':'normal', 'size':20}
        ax = plt.figure(figsize=(8, 6))
        ax = sns.heatmap(log['wPFC2MD_list'][-1], cmap='Reds')
        ax.set_xticks([0, 255])
        ax.set_xticklabels([1, 256], rotation=0)
        ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
        ax.set_xlabel('PFC neuron index', fontdict=font)
        ax.set_ylabel('MD neuron index', fontdict=font)
        ax.set_title('wPFC2MD', fontdict=font)
        cbar = ax.collections[0].colorbar
        cbar.set_label('connection weight', fontdict=font)
        plt.tight_layout()
        # plt.savefig('./animation/'+'wPFC2MD.png')
        plt.show()

        # Heatmap wMD2PFC
        font = {'family':'Times New Roman','weight':'normal', 'size':20}
        ax = plt.figure(figsize=(8, 6))
        ax = sns.heatmap(log['wMD2PFC_list'][-1], cmap='Blues_r')
        ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
        ax.set_yticks([0, 255])
        ax.set_yticklabels([1, 256], rotation=0)
        ax.set_xlabel('MD neuron index', fontdict=font)
        ax.set_ylabel('PFC neuron index', fontdict=font)
        ax.set_title('wMD2PFC', fontdict=font)
        cbar = ax.collections[0].colorbar
        cbar.set_label('connection weight', fontdict=font)
        plt.tight_layout()
        # plt.savefig('./animation/'+'wMD2PFC.png')
        plt.show()

# recurrent weights
if False:
    font = {'family':'Times New Roman','weight':'normal', 'size':15}
    plt.figure()
    ms = plt.matshow(net.rnn.h2h.weight.cpu().detach(), cmap='Reds')
    plt.xlabel('PFC neuron index', fontdict=font)
    plt.ylabel('PFC neuron index', fontdict=font)
    plt.xticks(ticks=[0, 255], labels=[1, 256])
    plt.yticks(ticks=[0, 255], labels=[1, 256])
    plt.title('Recurrent weights', fontdict=font)
    plt.colorbar(ms)
    plt.show()

# PFC+MD VS PFC+EWC, PFC
if False:
    FILE_PATH = './files/comparison/dropout/'
    # tasks, TASK_NAME = ['yang19.go-v0', 'yang19.rtgo-v0'], 'gortgo'
    tasks, TASK_NAME = ['yang19.dms-v0', 'yang19.dmc-v0'], 'dmsdmc'
    # tasks, TASK_NAME = ['yang19.dnms-v0', 'yang19.dnmc-v0'], 'dnmsdnmc'
    # tasks, TASK_NAME = ['yang19.dlygo-v0', 'yang19.dnmc-v0'], 'dlygodnmc'
    settings = ['withMD', 'PFCEWC', 'noMD']
    colors = ['red', 'green', 'black']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC']
    linewidths = [2, 1, 1]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    for env_id in range(len(tasks)):
        plt.figure()
        for i in range(len(settings)):
            PATH = FILE_PATH + 'log_' + TASK_NAME + '_' + settings[i] + '.npy'
            log = np.load(PATH, allow_pickle=True).item()
            plt.plot(log['stamps'], log['act_perfs'][env_id], linewidth=linewidths[i], color=colors[i], label=labels[i])
            plt.fill_between(x=[   0,  20000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.fill_between(x=[20000, 40000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.02)
            plt.fill_between(x=[40000, 50000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.legend(bbox_to_anchor = (1.3, 0.7), prop=legend_font)
            plt.xlabel('Trials', fontdict=label_font)
            plt.ylabel('Performance', fontdict=label_font)
            plt.title('Task{:d}: '.format(env_id+1)+tasks[env_id], fontdict=title_font)
            plt.xlim([0.0, None])
            plt.ylim([0.0, 1.01])
            plt.yticks([0.1*i for i in range(11)])
            plt.tight_layout()
        plt.show()

# PFC+MD VS CTRNN+EWC, CTRNN+MD, CTRNN
if False:
    FILE_PATH = './files/comparison/dropout/'
    # tasks, TASK_NAME = ['yang19.go-v0', 'yang19.rtgo-v0'], 'gortgo'
    # tasks, TASK_NAME = ['yang19.dms-v0', 'yang19.dmc-v0'], 'dmsdmc'
    # tasks, TASK_NAME = ['yang19.dnms-v0', 'yang19.dnmc-v0'], 'dnmsdnmc'
    tasks, TASK_NAME = ['yang19.dlygo-v0', 'yang19.dnmc-v0'], 'dlygodnmc'
    settings = ['withMD', 'CTRNNEWC', 'CTRNNMD', 'CTRNN']
    colors = ['red', 'blue', 'green', 'black']
    labels = ['PFC+MD', 'CTRNN+EWC', 'CTRNN+MD', 'CTRNN']
    linewidths = [2, 1, 1, 1]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    for env_id in range(len(tasks)):
        plt.figure()
        for i in range(len(settings)):
            PATH = FILE_PATH + 'log_' + TASK_NAME + '_' + settings[i] + '.npy'
            log = np.load(PATH, allow_pickle=True).item()
            plt.plot(log['stamps'], log['act_perfs'][env_id], linewidth=linewidths[i], color=colors[i], label=labels[i])
            plt.fill_between(x=[   0,  20000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.fill_between(x=[20000, 40000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.02)
            plt.fill_between(x=[40000, 50000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.legend(bbox_to_anchor = (1.4, 0.7), prop=legend_font)
            plt.xlabel('Trials', fontdict=label_font)
            plt.ylabel('Performance', fontdict=label_font)
            plt.title('Task{:d}: '.format(env_id+1)+tasks[env_id], fontdict=title_font)
            plt.xlim([0.0, None])
            plt.ylim([0.0, 1.01])
            plt.yticks([0.1*i for i in range(11)])
            plt.tight_layout()
        plt.show()

# compare task pairs in one figure
## compute averages of performance
if False:
    FILE_PATH = './files/comparison/dropout/'
    # TASK_NAMES, setting = ['dnmsdnmc', 'dmsdmc', 'dmsdm2', 'dmsdm1', 'dmsctxdm1', 'dlygodnmc', 'dlyantidnms', 'ctxdm1dms'], 'withMD'
    # TASK_NAMES, setting = ['dnmsdnmc', 'dmsdmc', 'dlygodnmc', 'dlyantidnms'], 'withMD'
    # TASK_NAMES, setting = ['dnmsdnmc', 'dmsdmc', 'dlygodnmc', 'dlyantidnms'], 'noMD'
    # TASK_NAMES, setting = ['dnmsdnmc', 'dmsdmc', 'dlygodnmc'], 'PFCEWC'
    for i in range(len(TASK_NAMES)):
        TASK_NAME = TASK_NAMES[i]
        PATH = FILE_PATH + 'log_' + TASK_NAME + '_' + setting + '.npy'
        log = np.load(PATH, allow_pickle=True).item()
        if i == 0:
            act_perfs = np.array(log['act_perfs'])
        else:
            act_perfs += np.array(log['act_perfs'])
    act_perfs = act_perfs/len(TASK_NAMES)
    np.save('./files/'+'avg_perfs_'+setting+'.npy', act_perfs)
## PFC+MD VS PFC+EWC, PFC
if False:
    FILE_PATH = './files/comparison/average_perfs/'
    settings = ['withMD', 'PFCEWC', 'noMD']
    colors = ['red', 'green', 'black']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC']
    linewidths = [2, 1, 1]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    for env_id in range(2): # 2 tasks
        plt.figure()
        for i in range(len(settings)):
            PATH = FILE_PATH + 'avg_perfs_' + settings[i] + '.npy'
            act_perfs = np.load(PATH)
            plt.plot(act_perfs[env_id, :], linewidth=linewidths[i], color=colors[i], label=labels[i])
            plt.fill_between(x=[ 0,  40] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.fill_between(x=[40,  80] , y1=0.0, y2=1.01, facecolor='green', alpha=0.02)
            plt.fill_between(x=[80, 100] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.legend(bbox_to_anchor = (1.3, 0.7), prop=legend_font)
            plt.xlabel('Trials', fontdict=label_font)
            plt.ylabel('Performance', fontdict=label_font)
            plt.title('Task{:d}'.format(env_id+1), fontdict=title_font)
            plt.xlim([0.0, None])
            plt.ylim([0.0, 1.01])
            plt.yticks([0.1*i for i in range(11)])
            plt.tight_layout()
        plt.show()

# scale up test
## compute averages of performance
if True:
    FILE_PATH = './files/two_PFCs/default_test1/'
    # FILE_PATH = './files/scaleup/noMD/'

    settings = ['withMD', 'PFCEWC', 'noMD']
    # settings = ['withMD', 'noMD']
    # settings = ['withMD']
    # settings = ['noMD']
    
    ITER = list(range(40))
    LEN = len(ITER)
    for setting in settings:
        for i in ITER:
            PATH = FILE_PATH + str(i) + '_log_' + setting + '.npy'
            log = np.load(PATH, allow_pickle=True).item()
            if i == 0:
                act_perfs = np.array(log['act_perfs'])
            else:
                act_perfs += np.array(log['act_perfs'])
        time_stamps = log['stamps']
        act_perfs = act_perfs/LEN
        np.save('./files/'+'avg_perfs_'+setting+'.npy', act_perfs)
        np.save('./files/'+'time_stamps_'+setting+'.npy', time_stamps)
## PFC+MD VS PFC+EWC, PFC
if True:
    FILE_PATH = './files/'
    settings = ['withMD', 'PFCEWC', 'noMD']
    colors = ['red', 'blue', 'black']
    labels = ['PFC+MD', 'PFC+EWC', 'PFC']
    linewidths = [2, 2, 2]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    for env_id in range(2): # 2 tasks
        plt.figure()
        for i in range(len(settings)):
            act_perfs = np.load(FILE_PATH + 'avg_perfs_' + settings[i] + '.npy')
            time_stamps = np.load(FILE_PATH + 'time_stamps_' + settings[i] + '.npy')
            plt.plot(time_stamps, act_perfs[env_id, :], linewidth=linewidths[i], color=colors[i], label=labels[i])
            plt.fill_between(x=[    0, 20000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.fill_between(x=[20000, 40000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.02)
            plt.fill_between(x=[40000, 50000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.02)
            plt.legend(bbox_to_anchor = (1.35, 0.65), prop=legend_font)
            plt.xlabel('Trials', fontdict=label_font)
            plt.ylabel('Performance', fontdict=label_font)
            plt.title('Task{:d}'.format(env_id+1), fontdict=title_font)
            plt.xlim([0.0, None])
            plt.ylim([0.0, 1.01])
            plt.yticks([0.1*i for i in range(11)])
            plt.tight_layout()
        plt.show()