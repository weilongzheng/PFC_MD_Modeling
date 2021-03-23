import pickle
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from temp_task import RihkyeTask
from temp_model import FullNetwork

Ntrain = 10

dataset = RihkyeTask(Ntrain=Ntrain, Ntasks=2, blockTrain=True)

n_time = 200
n_neuron = 1000
n_neuron_per_cue = 200
Num_MD = 20
num_active = 10  # num MD active per context
n_output = 2
model = FullNetwork(n_neuron, n_neuron_per_cue, Num_MD, num_active,
                    MDeffect=False)

log = defaultdict(list)

mses = list()
for i in range(3*Ntrain):
    print(i)
    input, target = dataset()
    output = model(input, target)
    mse = np.mean((output - target)**2)
    log['mse'].append(mse)
    # log['model.md.wPFC2MD.norm'] = np.linalg.norm(model.md.wPFC2MD)

filename = Path('files') / 'tmp'
os.makedirs(filename, exist_ok=True)
with open(filename / 'log.pkl', 'wb') as f:
    pickle.dump(log, f)
