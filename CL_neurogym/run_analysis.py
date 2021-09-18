import numpy as np
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
## compute averages of performance
if 0:
    # FILE_PATH = './files/scaleup_twotasks_1/baselines/'
    FILE_PATH = './files/scaleup_threetasks_2/noisestd0dot01/'

    # settings = ['EWC', 'SI', 'PFC']
    settings = ['PFCMD']

    ITER = list(range(36))
    LEN = len(ITER)
    for setting in settings:
        for i in ITER:
            PATH = FILE_PATH + str(i) + '_log_' + setting + '.npy'
            log = np.load(PATH, allow_pickle=True).item()
            if i == 0:
                act_perfs = np.array(log.act_perfs)
            else:
                act_perfs += np.array(log.act_perfs)
        time_stamps = log.stamps
        act_perfs = act_perfs/LEN
        np.save('./files/'+'avg_perfs_'+setting+'noisestd0dot01.npy', act_perfs)
        np.save('./files/'+'time_stamps_'+setting+'noisestd0dot01.npy', time_stamps)
## PFC+MD VS PFC+EWC, PFC
if 1:
    FILE_PATH = './files/scaleup_threetasks_2/'
    # settings = ['PFCMDnoisestd0', 'PFCMDnoisestd0dot01', 'PFCMDnoisestd0dot1', 'EWC', 'SI', 'PFC']
    # colors = ['red', 'orange', 'green', 'blue', 'purple', 'black']
    # labels = ['PFC+MD ($\sigma_{noise}=0$)', 'PFC+MD ($\sigma_{noise}=0.01$)', 'PFC+MD ($\sigma_{noise}=0.1$)', 'PFC+EWC', 'PFC+SI', 'PFC']
    # linewidths = [3, 3, 3, 2, 2, 2]
    settings = ['PFCMDnoisestd0dot01', 'EWC', 'SI', 'PFC']
    colors = ['red', 'blue', 'purple', 'black']
    labels = ['PFC+MD ($\sigma_{noise}=0.01$)', 'PFC+EWC', 'PFC+SI', 'PFC']
    linewidths = [3, 2, 2, 2]
    label_font = {'family':'Times New Roman','weight':'normal', 'size':15}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    for env_id in range(3): # 2 tasks
        plt.figure(figsize=(9, 6))
        for i in range(len(settings)):
            act_perfs = np.load(FILE_PATH + 'avg_perfs_' + settings[i] + '.npy')
            time_stamps = np.load(FILE_PATH + 'time_stamps_' + settings[i] + '.npy')
            plt.plot(time_stamps, act_perfs[env_id, :], linewidth=linewidths[i], color=colors[i], label=labels[i])
            plt.fill_between(x=[    0, 20000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.01)
            plt.fill_between(x=[20000, 40000] , y1=0.0, y2=1.01, facecolor='green', alpha=0.01)
            # plt.fill_between(x=[40000, 50000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.01)
            plt.fill_between(x=[40000, 60000] , y1=0.0, y2=1.01, facecolor='purple', alpha=0.01)
            plt.fill_between(x=[60000, 70000] , y1=0.0, y2=1.01, facecolor='red', alpha=0.01)
            plt.legend(bbox_to_anchor = (1.35, 0.65), prop=legend_font)
            plt.xlabel('Trials', fontdict=label_font)
            plt.ylabel('Performance', fontdict=label_font)
            plt.title('Task{:d}'.format(env_id+1), fontdict=title_font)
            plt.xlim([0.0, None])
            plt.ylim([0.0, 1.01])
            plt.yticks([0.1*i for i in range(11)])
            plt.tight_layout()
        plt.show()