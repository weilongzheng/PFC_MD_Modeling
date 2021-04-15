'''
Pytorch built-in Elman RNN + MD layer
'''

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
import sys
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')

from temp_task import RikhyeTaskBatch
from temp_model import Elman_MD



#---------------- Rikhye dataset with batch dimension ----------------#
RNGSEED = 5
np.random.seed([RNGSEED])

num_cueingcontext = 2
num_cue = 2
num_rule = 2
rule = [0, 1, 0, 1]
#blocklen = [500, 500, 200]
blocklen = [50, 50, 50]
block_cueingcontext = [0, 1, 0]
tsteps = 200
cuesteps = 100
batch_size = 1

dataset = RikhyeTaskBatch(num_cueingcontext=num_cueingcontext, num_cue=num_cue, num_rule=num_rule,\
                          rule=rule, blocklen=blocklen, block_cueingcontext=block_cueingcontext,\
                          tsteps=tsteps, cuesteps=cuesteps, batch_size=batch_size)


#---------------- Elman_MD model ----------------#
input_size = 4 # 4 cues
hidden_size = 200 # number of PFC neurons
output_size = 2 # 2 rules
num_layers = 1
nonlinearity = 'tanh'
Num_MD = 10
num_active = 5
MDeffect = True
log = dict()
log['mse'] = []

model = Elman_MD(input_size=input_size, hidden_size=hidden_size, output_size=output_size,\
                 num_layers=num_layers, nonlinearity=nonlinearity, Num_MD=Num_MD, num_active=num_active,\
                 tsteps=tsteps, MDeffect=MDeffect)

print(model)
for param in model.named_parameters():
    print(param[0], param[1].shape)
print('\n', end='')

#---------------- Training ----------------#
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

total_step = sum(blocklen)//batch_size
print_step = 10
running_loss = 0.0
running_train_time = 0


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
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # normalization
    optimizer.step()

    # print statistics
    mse = loss.item()
    log['mse'].append(mse)

    running_train_time += time.time() - train_time_start
    running_loss += loss.item()
    if i % print_step == (print_step - 1):

        print('Total step: {:d}'.format(total_step))
        print('Training sample index: {:d}-{:d}'.format(i+1-print_step, i+1))

        # running loss
        print('loss: {:0.5f}'.format(running_loss / print_step))
        running_loss = 0.0

        # training time
        print('Predicted left training time: {:0.0f} s'.format(
        (running_train_time) * (total_step - i - 1) / print_step),
        end='\n\n')
        running_train_time = 0

        # save model during training
        if  MDeffect == True:  
            log['wPFC2MD'] = model.md.wPFC2MD
            log['wMD2PFC'] = model.md.wMD2PFC
            log['wMD2PFCMult'] = model.md.wMD2PFCMult

        filename = Path('files')
        os.makedirs(filename, exist_ok=True)
        file_training = 'test_Elman_MD'+'_R'+str(RNGSEED)+'.pkl'
        with open(filename / file_training, 'wb') as f:
            pickle.dump(log, f)


print('Finished Training')



#---------------- Make some plots ----------------#
with open(filename / file_training, 'rb') as f:
    log = pickle.load(f)

# Plot MSE curve
plt.plot(log['mse'], label='Elman MD')
plt.xlabel('Cycles')
plt.ylabel('MSE loss')
plt.legend()
#plt.xticks([0, 500, 1000, 1200])
#plt.ylim([0.0, 1.0])
#plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#plt.tight_layout()
plt.show()