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


log = np.load('./files/'+'log_withMD.npy', allow_pickle=True).item()


# PFC outputs within a cycle
if False:
    PFCouts_all = log['PFCouts_all']
    print(PFCouts_all.shape) # (num_cycles, seq_len, batch_size, hidden_size)
    font = {'family':'Times New Roman','weight':'normal', 'size':25}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    idx_cycle = 400
    PFCouts_cycle = log['PFCouts_all'][idx_cycle, :, 0, :]
    for i in range(PFCouts_cycle.shape[0]):
        meanPFCouts_cycle = np.mean(PFCouts_cycle[i, :])
        plt.plot(PFCouts_cycle[i, :])
        plt.axhline(y=meanPFCouts_cycle, color='r', linestyle='-')
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
if True:
    MDpreTraces_all = log['MDpreTraces_all']
    MDpreTrace_threshold_all = log['MDpreTrace_threshold_all']
    font = {'family':'Times New Roman','weight':'normal', 'size':25}
    label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    # idx_cycle = 400
    for idx_cycle in [0, 1, 50, 199, 200, 250, 399, 400]:
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

# if config['MDeffect']:
#     # Heatmap wPFC2MD
#     font = {'family':'Times New Roman','weight':'normal', 'size':20}
#     ax = plt.figure(figsize=(8, 6))
#     ax = sns.heatmap(log['wPFC2MD_list'][-1], cmap='Reds')
#     ax.set_xticks([0, 255])
#     ax.set_xticklabels([1, 256], rotation=0)
#     ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
#     ax.set_xlabel('PFC neuron index', fontdict=font)
#     ax.set_ylabel('MD neuron index', fontdict=font)
#     ax.set_title('wPFC2MD', fontdict=font)
#     cbar = ax.collections[0].colorbar
#     cbar.set_label('connection weight', fontdict=font)
#     plt.tight_layout()
#     # plt.savefig('./animation/'+'wPFC2MD.png')
#     plt.show()

#     # Heatmap wMD2PFC
#     font = {'family':'Times New Roman','weight':'normal', 'size':20}
#     ax = plt.figure(figsize=(8, 6))
#     ax = sns.heatmap(log['wMD2PFC_list'][-1], cmap='Blues_r')
#     ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
#     ax.set_yticks([0, 255])
#     ax.set_yticklabels([1, 256], rotation=0)
#     ax.set_xlabel('MD neuron index', fontdict=font)
#     ax.set_ylabel('PFC neuron index', fontdict=font)
#     ax.set_title('wMD2PFC', fontdict=font)
#     cbar = ax.collections[0].colorbar
#     cbar.set_label('connection weight', fontdict=font)
#     plt.tight_layout()
#     # plt.savefig('./animation/'+'wMD2PFC.png')
#     plt.show()

# recurrent weights
# font = {'family':'Times New Roman','weight':'normal', 'size':15}
# plt.figure()
# ms = plt.matshow(net.rnn.h2h.weight.cpu().detach(), cmap='Reds')
# plt.xlabel('PFC neuron index', fontdict=font)
# plt.ylabel('PFC neuron index', fontdict=font)
# plt.xticks(ticks=[0, 255], labels=[1, 256])
# plt.yticks(ticks=[0, 255], labels=[1, 256])
# plt.title('Recurrent weights', fontdict=font)
# plt.colorbar(ms)
# plt.show()