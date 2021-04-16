import pickle
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from task import RikhyeTask
from model import FullNetwork

RNGSEED = 5
np.random.seed([RNGSEED])

Ntrain = 500             # number of training cycles for each context
Nextra = 200            # add cycles to show if block1
Ncontexts = 2
inpsPerConext = 2
dataset = RikhyeTask(Ntrain=Ntrain, Nextra = Nextra, Ncontexts=Ncontexts, inpsPerConext = inpsPerConext, blockTrain=True)

n_neuron = 1000
n_neuron_per_cue = 200
Num_MD = 10
num_active = 5  # num MD active per context
n_output = 2
MDeffect = True
model = FullNetwork(n_neuron, n_neuron_per_cue, Num_MD, num_active, MDeffect=MDeffect)

log = defaultdict(list)

num_cycle_train = Ntrain*Ncontexts+Nextra
mses = list()
MDpreTraces = np.zeros(shape=(num_cycle_train,n_neuron))
MDouts_all = np.zeros(shape=(num_cycle_train,Num_MD))
PFCouts_all = np.zeros(shape=(num_cycle_train,n_neuron))
for i in range(num_cycle_train):
    print('training '+str(i))
    input, target = dataset()
    output = model(input, target)
    mse = np.mean((output - target)**2)*Ncontexts # one cycle has Ncontexts

#    mse = np.mean((output[:200] - target[:200])**2)
#    mse += np.mean((output[200:] - target[200:])**2)
    PFCouts_all[i,:] = model.pfc.activity
    log['mse'].append(mse)
    if  MDeffect == True:
        MDouts_all[i,:] = model.md_output
        MDpreTraces[i,:] = model.md.MDpreTrace
    
if  MDeffect == True:  
    log['wPFC2MD'] = model.md.wPFC2MD
    log['wMD2PFC'] = model.md.wMD2PFC
    log['wMD2PFCMult'] = model.md.wMD2PFCMult

filename = Path('files') #/ 'tmp'
os.makedirs(filename, exist_ok=True)
file_training = 'train_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_R'+str(RNGSEED)+'.pkl'
with open(filename / file_training, 'wb') as f:
    pickle.dump(log, f)
    
    
## Testing
Ntest = 500
Nextra = 0
tsteps = 200
test_set = RikhyeTask(Ntrain=Ntest, Nextra = Nextra, Ncontexts=Ncontexts, inpsPerConext = inpsPerConext, blockTrain=False)

log = defaultdict(list)

num_cycle_test = Ntest+Nextra
mses = list()
cues_all = np.zeros(shape=(num_cycle_test*inpsPerConext*Ncontexts,tsteps, inpsPerConext*Ncontexts))
MDouts_all = np.zeros(shape=(num_cycle_test*inpsPerConext*Ncontexts,tsteps,Num_MD))
PFCouts_all = np.zeros(shape=(num_cycle_test*inpsPerConext*Ncontexts,tsteps,n_neuron))
for i in range(num_cycle_test):
    print('testing '+str(i))
    input, target = test_set()
    output = model(input, target)
    
    tstart = 0
    for itrial in range(inpsPerConext*Ncontexts): 
        PFCouts_all[i*inpsPerConext*Ncontexts+tstart,:,:] = model.pfc_output_t[tstart*tsteps:(tstart+1)*tsteps,:]
        MDouts_all[i*inpsPerConext*Ncontexts+tstart,:,:] = model.md_output_t[tstart*tsteps:(tstart+1)*tsteps,:]
        cues_all[i*inpsPerConext*Ncontexts+tstart,:,:] = input[tstart*tsteps:(tstart+1)*tsteps,:]
        tstart += 1
    #import pdb;pdb.set_trace()    
    mse = np.mean((output - target)**2)*inpsPerConext*Ncontexts # one cycle has 4 cues
    log['mse'].append(mse)

log['wPFC2MD'] = model.md.wPFC2MD
log['wMD2PFC'] = model.md.wMD2PFC
log['wMD2PFCMult'] = model.md.wMD2PFCMult
    
filename = Path('files') #/ 'tmp'
os.makedirs(filename, exist_ok=True)
file_training = 'test_numMD'+str(Num_MD)+'_numContext'+str(Ncontexts)+'_MD'+str(MDeffect)+'_R'+str(RNGSEED)+'.pkl'
with open(filename / file_training, 'wb') as f:
    pickle.dump({'log':log,'PFCouts_all':PFCouts_all,'MDouts_all':MDouts_all,'cues_all':cues_all}, f, protocol = 4)

