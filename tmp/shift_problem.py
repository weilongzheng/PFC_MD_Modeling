'''
Fix PFC and shift wIn every cycle and check network performance
'''


import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
import os
import sys
import pickle
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from task import RikhyeTask
from model import PytorchPFCMD
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from pygifsicle import optimize


# Generate trainset
#  set random seed
RNGSEED = 5 # default 5
np.random.seed([RNGSEED])
torch.manual_seed(RNGSEED)

Ntrain = 800            # number of training cycles for each context; default 200
Nextra = 0            # add cycles to show if block1; default 200; if 0, no switch back to past context
Ncontexts = 2           # number of cueing contexts (e.g. auditory cueing context)
inpsPerConext = 2       # in a cueing context, there are <inpsPerConext> kinds of stimuli
                         # (e.g. auditory cueing context contains high-pass noise and low-pass noise)
dataset = RikhyeTask(Ntrain=Ntrain, Nextra=Nextra, Ncontexts=Ncontexts, inpsPerConext=inpsPerConext, blockTrain=True)

# Model settings
n_neuron = 1000
n_neuron_per_cue = 200
Num_MD = 10
num_active = 5  # num MD active per context
n_output = 2
MDeffect = True
PFClearn = False
shift_list = [0] # shift step list

for shift in shift_list:


    model = PytorchPFCMD(Num_PFC=n_neuron, n_neuron_per_cue=n_neuron_per_cue, Num_MD=Num_MD, num_active=num_active, num_output=n_output, \
    MDeffect=MDeffect)

    # Training
    criterion = nn.MSELoss()
    training_params = list()
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        print(name)
        training_params.append(param)

    if PFClearn==True:
        print('pfc.Jrec')
        print('\n', end='')
        training_params.append(model.pfc.Jrec)
    else:
        print('\n', end='')
        
        
    Jrec_init = model.pfc.Jrec.clone()#.numpy()
    #print(Jrec_init) # debug
    optimizer = torch.optim.Adam(training_params, lr=1e-3)
    #import pdb;pdb.set_trace()

    #total_step = Ntrain*Ncontexts+Nextra
    total_step = Ntrain+Nextra
    print_step = 10 # print statistics every print_step
    save_W_step = 10 # save wPFC2MD and wMD2PFC every save_W_step
    running_loss = 0.0
    running_train_time = 0
    mses = list()
    #losses = []
    #timestamps = []
    #model_name = 'model-' + str(int(time.time()))
    #savemodel = False
    log = defaultdict(list)
    MDpreTraces = np.zeros(shape=(total_step,n_neuron))
    #MDouts_all = np.zeros(shape=(total_step,Num_MD))
    #PFCouts_all = np.zeros(shape=(total_step,n_neuron))
    tsteps = 200
    MDouts_all = np.zeros(shape=(total_step*inpsPerConext,tsteps,Num_MD))
    PFCouts_all = np.zeros(shape=(total_step*inpsPerConext,tsteps,n_neuron))


    for i in range(total_step):

        train_time_start = time.time()

        # extract data
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float)
        labels = torch.from_numpy(labels).type(torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, labels)
        # check PFC and MD outputs
        # PFCouts_all[i,:] = model.pfc.activity.detach().numpy()
        # if  MDeffect == True:
        #     MDouts_all[i,:] = model.md_output
        #     MDpreTraces[i,:] = model.md.MDpreTrace
        tstart = 0
        for itrial in range(inpsPerConext): 
            PFCouts_all[i*inpsPerConext+tstart,:,:] = model.pfc_outputs.detach().numpy()[tstart*tsteps:(tstart+1)*tsteps,:]
            if  MDeffect == True:
                MDouts_all[i*inpsPerConext+tstart,:,:] = model.md_output_t[tstart*tsteps:(tstart+1)*tsteps,:]
            tstart += 1

        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the norm of gradients 
        if PFClearn==True:
            torch.nn.utils.clip_grad_norm_(model.pfc.Jrec, 1e-6) # clip the norm of gradients; Jrec 1e-6
        optimizer.step()
        
        #import pdb;pdb.set_trace()

        # shift wIn every training cycle here
        # if i in [149]:
        #     model.sensory2pfc.shift(shift=shift)
        ####print(model.sensory2pfc.wIn[:, 0]) # debug

        # print statistics
        mse = loss.item()
        log['mse'].append(mse)
        running_train_time += time.time() - train_time_start
        running_loss += loss.item()

        #  save wPFC2MD and wMD2PFC
        if i % save_W_step == (save_W_step - 1):
            log['wPFC2MD_list'].append(model.md.wPFC2MD)
            log['wMD2PFC_list'].append(model.md.wMD2PFC)

        if i % print_step == (print_step - 1):

            print('Total step: {:d}'.format(total_step))
            print('Training sample index: {:d}-{:d}'.format(i+1-print_step, i+1))
            print('shift: {:d}'.format(shift))

            # running loss
            print('loss: {:0.5f}'.format(running_loss / print_step))
            #losses.append(running_loss / print_step)
            #timestamps.append(i+1-print_step)
            running_loss = 0.0

            # training time
            print('Predicted left training time: {:0.0f} s'.format(
                (running_train_time) * (total_step - i - 1) / print_step),
                end='\n\n')
            running_train_time = 0
            #print(model.pfc.Jrec)

            # if savemodel:
            #     # save model every print_step
            #     fname = os.path.join('models', model_name + '.pt')
            #     torch.save(model.state_dict(), fname)

            #     # save info of the model
            #     fpath = os.path.join('models', model_name + '.txt')
            #     with open(fpath, 'w') as f:
            #         f.write('input_size = ' + str(input_size) + '\n')
            #         f.write('hidden_size = ' + str(hidden_size) + '\n')
            #         f.write('output_size = ' + str(output_size) + '\n')
            #         f.write('num_layers = ' + str(num_layers) + '\n')

            # save model during training
            log['Jrec'] = model.pfc.Jrec
            if  MDeffect == True:  
                log['wPFC2MD'] = model.md.wPFC2MD
                log['wMD2PFC'] = model.md.wMD2PFC
                log['wMD2PFCMult'] = model.md.wMD2PFCMult

            filename = Path('files')
            os.makedirs(filename, exist_ok=True)
            # file_training = 'Ntrain'+str(Ntrain)+'_Nextra'+str(Nextra)+'_train_numMD'+str(Num_MD)+\
            #                 '_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+\
            #                 '_shift'+str(shift)+'_R'+str(RNGSEED)+'.pkl'
            file_training = 'Animation'+'Ntrain'+str(Ntrain)+'_Nextra'+str(Nextra)+'_train_numMD'+str(Num_MD)+\
                            '_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_PFC'+str(PFClearn)+\
                            '_shift'+str(shift)+'_R'+str(RNGSEED)+'.pkl'
            with open(filename / file_training, 'wb') as f:
                pickle.dump(log, f)


    print('Finished Training')

    # Make some plots
    with open(filename / file_training, 'rb') as f:
        log = pickle.load(f)

    # Plot MSE curve
    plt.plot(log['mse'], label=f'With MD; shift step = {shift}')
    plt.xlabel('Cycles')
    plt.ylabel('MSE loss')
    plt.legend()
    #plt.xticks([0, 500, 1000, 1200])
    #plt.ylim([0.0, 1.0])
    #plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    plt.show()

    ## plot pfc2md weights
    # wPFC2MD = log['wPFC2MD']
    # number = Num_MD
    # cmap = plt.get_cmap('rainbow') 
    # colors = [cmap(i) for i in np.linspace(0,1,number)]
    # plt.figure(figsize=(18,20))
    # for i,color in enumerate(colors, start=1):
    #     plt.subplot(number,1,i)
    #     plt.plot(wPFC2MD[i-1,:],color=color)
    # plt.xlabel(f'wPFC2MD; shift step = {shift}')
    # #plt.suptitle(f'wPFC2MD; shift step = {shift}')
    # plt.show()

    ## plot md2pfc weights
    # wMD2PFC = log['wMD2PFC']
    # number = Num_MD
    # cmap = plt.get_cmap('rainbow') 
    # colors = [cmap(i) for i in np.linspace(0,1,number)]
    # plt.figure(figsize=(18, 20))
    # for i,color in enumerate(colors, start=1):
    #     plt.subplot(number,1,i)
    #     plt.plot(wMD2PFC[:,i-1],color=color)
    # plt.xlabel(f'wMD2PFC; shift step = {shift}')
    # #plt.suptitle(f'wMD2PFC; shift step = {shift}')
    # plt.show()

    ## plot pfc recurrent weights before and after training
    #fig, axes = plt.subplots(nrows=1, ncols=2)
    #Jrec = model.pfc.Jrec.detach().numpy()
    #Jrec_init = Jrec_init
    ## find minimum of minima & maximum of maxima
    #minmin = np.min([np.min(Jrec_init), np.min(Jrec)])
    #maxmax = np.max([np.max(Jrec_init), np.max(Jrec)])
    #num_show = 200
    #im1 = axes[0].imshow(Jrec_init[:num_show,:num_show], vmin=minmin, vmax=maxmax,
    #                     extent=(0,num_show,0,num_show), cmap='viridis')
    #im2 = axes[1].imshow(Jrec[:num_show,:num_show], vmin=minmin, vmax=maxmax,
    #                     extent=(0,num_show,0,num_show), cmap='viridis')
    #
    ## add space for colour bar
    #fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    #fig.colorbar(im2, cax=cbar_ax)
