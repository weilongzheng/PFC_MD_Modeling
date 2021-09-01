import os
import sys


class BaseConfig(object):
    def __init__(self):
        # system
        self.device = 'cpu'
        self.RNGSEED = 5
        self.ROOT_DIR = os.getcwd()
        
        # dataset
        # self.task_seq = ['yang19.go-v0', 'yang19.rtgo-v0']
        # self.task_seq = ['yang19.dms-v0', 'yang19.dmc-v0']
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dnmc-v0']
        self.task_seq = ['yang19.dlyanti-v0', 'yang19.dnms-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dms-v0']
        # self.task_seq = ['yang19.rtgo-v0', 'yang19.ctxdm2-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dm1-v0']
        
        self.num_task = len(self.task_seq)
        self.env_kwargs = {'dt': 100}
        self.batch_size = 1

        # block training
        '''
        Customize the block training paradigm by changing self.task_seq, self.total_trials, self.switch_points, self.switch_taskid.
            e.g.
            To train four tasks serially for 10000 trials each:
            self.task_seq=[task1, task2, task3, task4]
            self.total_trials=40000
            self.switch_points=[0, 10000, 20000, 30000]
            self.switch_taskid=[0, 1, 2, 3]
        '''
        self.total_trials = 50000
        self.switch_points = [0, 20000, 40000]
        self.switch_taskid = [0, 1, 0]
        assert len(self.switch_points) == len(self.switch_taskid)

        # RNN model
        self.input_size = 33
        self.hidden_size = 400
        self.output_size = 17
        self.lr = 1e-4

        # test & plot
        self.test_every_trials = 500
        self.test_num_trials = 50
        self.plot_every_trials = 4000
    
    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class PFCMDConfig(BaseConfig):
    def __init__(self):
        super(PFCMDConfig, self).__init__()
        # PFC context
        self.hidden_ctx_size = 200
        self.sub_size = 100
        # MD
        self.MDeffect = True
        self.md_size = 4
        self.md_active_size = 2
        self.md_dt = 0.001

class PFCEWCConfig(BaseConfig):
    def __init__(self):
        super(PFCEWCConfig, self).__init__()
        # EWC
        self.EWC = True
        self.EWC_weight = 1e6
        self.EWC_num_trials = 1500

class PFCSIConfig(BaseConfig):
    def __init__(self):
        super(PFCSIConfig, self).__init__()
        # SI
        self.SI = True
        self.SI_c = 1e6
        self.c = self.SI_c
        self.SI_xi = 0.5
        self.xi = self.SI_xi
