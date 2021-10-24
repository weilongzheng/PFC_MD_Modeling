#!/usr/bin/env python
import os
from subprocess import call
import subprocess as sp
from time import sleep
import sys    

#%% Create the parameters for experiments in a table (expVars list of lists)

exp_name = 'pairs_corr_gates'
Var1 = list(range(30,60)) #[40, 9, 7,15,41, 8, 6,13,]  #15, 20, 30, 48]#range() # task pair number
Var2 = [ 0.5,  ] # gates_sparsity
Var3 = [0 , 1]# additive, mul
Var4 = [ 0.5] # gates std

# exp_name = 'batch_pairs'
# Var1 = range( 2)
# Var2 = [0.5, 1.0] #range(0,3, 1)
# Var3 = [30]#[0.1, 0.5, 1.] #range(30,91, 20)
# Var4 = [0, 1] #, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]


expVars =  [[x1, x2, x3, x4] for x1 in Var1 for x2 in Var2 for x3 in Var3 for x4 in Var4 ]
# [expVars[i].insert(0, i) for i in range(len(expVars))] # INSERT exp ID # in the first col.
print ('Total no of experiments generated : ', len(expVars))
# print(expVars)


#%% Load neuron
if sys.platform != 'win32':
    pass 
    # if slurm linux, can load any modules here.
    # sp.check_call("module load nrn/nrn-7.4", shell=True)  #Here it needs no splitting and shell=True
    

#%% Write the bash file to run an experiment

for par_set in expVars:
    sbatch_lines =["#!/bin/bash"
    ,  "#-------------------------------------------------------------------------------"
    , "#  SBATCH CONFIG"
    , "#-------------------------------------------------------------------------------"
    , "#SBATCH --nodes=1"
    , "#SBATCH -t 01:50:00"
    , "#SBATCH --gres=gpu:0"
    , "#SBATCH --mem=6G"
    # , '#SBATCH --output="showme.out"'
    , "#SBATCH --job-name=cl_{}".format(par_set[0])
    , "#-------------------------------------------------------------------------------"
    , "## Run model "]
    
    # args = ['-c "x%d=%g" ' % (i, par_set[i]) for i in range(len(par_set))]
    var1, var2, var3, var4 = par_set

    command_line = 'python run_PFCMD.py  {} 1 1 0 --var1={} --var2={} --var3={} --var4={}'.format(exp_name, var1, var2, var3, var4)  + f' --num_of_tasks=20'
    win_command_line = command_line + ' --os=windows'
    
    fsh = open('bash_generated.sh', 'w')
    fsh.write("\n".join(sbatch_lines))
    fsh.writelines(["\n", command_line])
    fsh.close()
    
    # Run the simulation
    if sys.platform == 'win32':
        sp.check_call(win_command_line.split(' '))
    else: #assuming Linux
        sp.check_call('sbatch bash_generated.sh'.split(' '))
    
    sleep(1)    # Inserts a 1 second pause
    
    