import pickle
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

filename = Path('files') / 'tmp'
os.makedirs(filename, exist_ok=True)
with open(filename / 'log.pkl', 'rb') as f:
    log = pickle.load(f)

plt.plot(log['mse'])