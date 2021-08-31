import os
import sys


class BaseConfig(object):
    def __init__(self):
        # system
        self.device = 'cpu'
        self.RNGSEED = 5
        self.ROOT_DIR = os.getcwd()
        
        # dataset
        self.taskpair = ['yang19.dlygo-v0', 'yang19.dnmc-v0']
        self.env_kwargs = {'dt': 100}
        self.batch_size = 1
        
        # RNN model
        self.input_size = 33
        self.hidden_size = 400
        self.sub_size = 200
        self.output_size = 17
        self.num_task = len(self.taskpair)
        self.lr = 1e-4

        # training & test
        self.total_trials = 50000
        self.switch_points = [0, 20000, 40000]
        self.switch_taskid = [0, 1, 0]
        assert len(self.switch_points) == len(self.switch_taskid)
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
        self.EWC_num_trials = 3000

class PFCSIConfig(BaseConfig):
    def __init__(self):
        super(PFCSIConfig, self).__init__()
        # SI
        self.SI = True
        self.SI_c = 1e6
        self.SI_xi = 0.5
