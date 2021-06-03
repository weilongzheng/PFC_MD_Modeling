# -*- coding: utf-8 -*-
# (c) Jun 2018 Aditya Gilra, EPFL., Uni-Bonn

import numpy as np
import matplotlib.pyplot as plt
import sys,shelve
import plot_utils as pltu
import matplotlib as mpl
import seaborn as sns
import pickle
MODELPATH = Path('./files')
FIGUREPATH = Path('./results')

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def plot_wPFC2MD(wPFC2MD):
    ax = plt.figure(figsize=(2.4,2))
    ax = sns.heatmap(wPFC2MD, cmap='Reds')
    ax.set_xticks([0, 399, 799, 999])
    ax.set_xticklabels([1, 400, 800, 1000], rotation=0)
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_xlabel('PFC Index')
    ax.set_ylabel('MD Index')
    cbar = ax.collections[0].colorbar
#    cbar.set_label('connection weight')
    plt.tight_layout()
    plt.show()
    
def plot_wMD2PFC(wMD2PFC):
    ax = plt.figure(figsize=(2.4,2))
    ax = sns.heatmap(wMD2PFC, cmap='Blues_r')
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_yticks([0, 399, 799, 999])
    ax.set_yticklabels([1, 400, 800, 1000], rotation=0)
    ax.set_xlabel('MD Index')
    ax.set_ylabel('PFC Index')
    cbar = ax.collections[0].colorbar
    plt.tight_layout()
    plt.show()
    
def plot_wMD2PFCMult(wMD2PFCMult):
    ax = plt.figure(figsize=(2.4,2))
    ax = sns.heatmap(wMD2PFCMult, cmap='Reds')
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rotation=0)
    ax.set_yticks([0, 399, 799, 999])
    ax.set_yticklabels([1, 400, 800, 1000], rotation=0)
    ax.set_xlabel('MD Index')
    ax.set_ylabel('PFC Index')
    cbar = ax.collections[0].colorbar
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    pickle_in = open('files/final/train_noiseJ_numMD10_numContext2_MDTrue_PFCFalse_R1.pkl','rb')
    data = pickle.load(pickle_in)
    
    wPFC2MD = data['wPFC2MD']
    wMD2PFC = data['wMD2PFC']
    wMD2PFCMult = data['wMD2PFCMult']
    
    plot_wPFC2MD(wPFC2MD)
    plt.savefig(FIGUREPATH/'wPFC2MD_noiseJ.pdf') 
    plot_wMD2PFC(wMD2PFC)
    plt.savefig(FIGUREPATH/'wMD2PFC_noiseJ.pdf') 
    import pdb;pdb.set_trace()
    plot_wMD2PFCMult(wMD2PFCMult)
    plt.savefig(FIGUREPATH/'wMD2PFCMult_noiseN.pdf') 
    
