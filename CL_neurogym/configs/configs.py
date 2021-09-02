import os
import sys
from models import MODES

'''
Class names of configs are based on the class names of models:
    Base -> BaseConfig
    EWC  -> EWCConfig
'''

class BaseConfig(object):
    def __init__(self):
        # system
        self.device = 'cpu'
        # self.device = 'cuda:0'
        self.RNGSEED = 5
        self.ROOT_DIR = os.getcwd()
        
        # dataset
        # 1. Two tasks
        # self.task_seq = ['yang19.go-v0', 'yang19.rtgo-v0']
        # self.task_seq = ['yang19.dms-v0', 'yang19.dmc-v0']
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dnms-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dms-v0']
        # self.task_seq = ['yang19.rtgo-v0', 'yang19.ctxdm2-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dm1-v0']
        # 2. Four tasks
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.dlygo-v0', 'yang19.go-v0']
        self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.anti-v0', 'yang19.dlyanti-v0']

        self.num_task = len(self.task_seq)
        self.env_kwargs = {'dt': 100}
        self.batch_size = 1

        # block training
        '''
        Customize the block training paradigm:
        1. Change self.task_seq, self.total_trials, self.switch_points.
            e.g.
            To train four tasks serially for 10000 trials each:
            self.task_seq=[task1, task2, task3, task4]
            self.total_trials=40000
            self.switch_points=[0, 10000, 20000, 30000]
        2. Change utils.get_task_id
        3. Change the task_ids of CL_model.end_task() in the run_baselines.py 
        '''
        self.total_trials = 50000
        self.switch_points = [0, 20000, 40000]
        self.switch_taskid = [0, 1, 0] # this config is deprecated right now
        assert len(self.switch_points) == len(self.switch_taskid)

        # RNN model
        self.input_size = 33
        self.hidden_size = 400
        self.output_size = 17
        self.lr = 1e-4

        # test
        self.test_every_trials = 500
        self.test_num_trials = 30

        # plot
        self.plot_every_trials = 4000
        self.save_fig = True

        # save variables
        self.FILEPATH = './files/'
        self.FILENAME = {
                        'config':    'config_PFC.npy',
                        'log':       'log_PFC.npy',
                        'plot_perf': 'performance_PFC_task.png',
        }
        # continual learning mode
        self.mode = None
    
    def set_task_seq(self, task_seq):
        self.task_seq = task_seq
        self.num_task = len(self.task_seq)
    
    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class PFCMDConfig(BaseConfig):
    def __init__(self):
        super(PFCMDConfig, self).__init__()
        # PFC context
        self.hidden_ctx_size = 400
        self.sub_size = 100
        # MD
        self.MDeffect = True
        self.md_size = 4
        self.md_active_size = 2
        self.md_dt = 0.001
        # save variables
        self.FILENAME = {
                        'config':    'config_PFCMD.npy',
                        'log':       'log_PFCMD.npy',
                        'plot_perf': 'performance_PFCMD_task.png',
        }

class EWCConfig(BaseConfig):
    def __init__(self):
        super(EWCConfig, self).__init__()
        # EWC
        self.EWC = True
        self.EWC_weight = 1e6
        self.EWC_num_trials = 1500
        # save variables
        self.FILENAME = {
                        'config':    'config_EWC.npy',
                        'log':       'log_EWC.npy',
                        'plot_perf': 'performance_EWC_task.png',
        }

class SIConfig(BaseConfig):
    def __init__(self):
        super(SIConfig, self).__init__()
        # SI
        self.SI = True
        self.SI_c = 1e6
        self.c = self.SI_c
        self.SI_xi = 0.5
        self.xi = self.SI_xi
        # save variables
        self.FILENAME = {
                        'config':    'config_SI.npy',
                        'log':       'log_SI.npy',
                        'plot_perf': 'performance_SI_task.png',
        }
