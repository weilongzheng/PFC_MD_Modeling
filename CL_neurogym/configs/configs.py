import os
import sys
from models import MODES
from configs.utils import get_similarity

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
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dnms-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dms-v0']
        # self.task_seq = ['yang19.rtgo-v0', 'yang19.ctxdm2-v0']
        self.task_seq = ['yang19.dlygo-v0', 'yang19.dnmc-v0'] # single task pair analysis
        # self.task_seq = ['yang19.dms-v0', 'yang19.dnms-v0']
        # self.task_seq = ['yang19.dms-v0', 'yang19.dmc-v0']
        # 2. Three tasks
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dm1-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dm1-v0', 'yang19.dlygo-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dm2-v0', 'yang19.dmc-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dm1-v0', 'yang19.dnmc-v0']
        # 3. Four tasks
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.dlygo-v0', 'yang19.go-v0']
        # self.task_seq = ['yang19.dms-v0', 'yang19.dnms-v0', 'yang19.dlygo-v0', 'yang19.go-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.go-v0', 'yang19.dmc-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.anti-v0', 'yang19.dlyanti-v0']
        self.human_task_names = ['{:<6}'.format(tn[7:-3]) for tn in self.task_seq] #removes yang19 and -v0

        self.num_task = len(self.task_seq)
        self.task_similarity = get_similarity(task_seq=self.task_seq) if self.num_task == 2 else None
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
        2. Change utils.get_task_seqs, utils.get_task_id
        3. Change PFCMD configs.
        4. Change the task_ids of CL_model.end_task() in the run_baselines.py & scaleup_baselines.py
        5. Change self.FILEPATH
        '''
        self.total_trials = 50000 # 70000
        self.switch_points = [0, 20000, 40000] # [0, 20000, 40000, 60000]
        self.switch_taskid = [0, 1, 0] # [0, 1, 2, 0] # this config is deprecated right now
        assert len(self.switch_points) == len(self.switch_taskid)

        # RNN model
        self.input_size = 33
        self.hidden_size = 600
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
                        'net':       'net_PFC.pt',
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

class PFCPFCctxConfig(BaseConfig):
    def __init__(self):
        super(PFCPFCctxConfig, self).__init__()
        # PFC context
        self.hidden_ctx_size = 400 # 450
        self.sub_size = 200 # 150
        self.sub_active_size = 50 # this config is deprecated right now
        self.sub_active_prob = 0.40
        self.hidden_ctx_noise = 0.01
        # save variables
        self.FILENAME = {
                        'config':    'config_PFCPFCctx.npy',
                        'log':       'log_PFCPFCctx.npy',
                        'net':       'net_PFCPFCctx.pt',
                        'plot_perf': 'performance_PFCPFCctx_task.png',
        }

class PFCMDConfig(BaseConfig):
    def __init__(self):
        super(PFCMDConfig, self).__init__()
        # PFC context
        self.hidden_ctx_size = 400 # 450
        self.sub_size = 200 # 150
        self.sub_active_size = 50 # this config is deprecated right now
        self.sub_active_prob = 0.40
        self.hidden_ctx_noise = 0.01
        # MD
        self.MDeffect = True
        self.md_size = 2 # 3
        self.md_active_size = 1
        self.md_dt = 0.001
        self.MDtoPFC_connect_prob = 1.00 # 0.10, overlapping MD to PFC effect # 1.00, disjoint MD to PFC effect
        # save variables
        self.FILENAME = {
                        'config':    'config_PFCMD.npy',
                        'log':       'log_PFCMD.npy',
                        'net':       'net_PFCMD.pt',
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
                        'net':       'net_EWC.pt',
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
                        'net':       'net_SI.pt',
                        'plot_perf': 'performance_SI_task.png',
        }

class SerialConfig(BaseConfig):
    def __init__(self, args= []):
        super(SerialConfig, self).__init__()
        # PFC context
        self.hidden_ctx_size = 400
        self.sub_size = 100
        # MD
        self.MDeffect = False
        self.md_size = 4
        self.md_active_size = 2
        self.md_dt = 0.001
        self.use_gates = False
        self.train_to_criterion = False
        self.same_rnn = True

        self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.dlygo-v0', 'yang19.go-v0']
        self.num_task = len(self.task_seq)
        self.env_kwargs = {'dt': 100}
        self.batch_size = 1

        # block training
        self.trials_per_task = 1000
        self.total_trials = int(self.num_task * self.trials_per_task)
        self.switch_points = list(range(0, self.total_trials, self.trials_per_task))
        self.switch_taskid = list(range(self.num_task) ) # this config is deprecated right now
        assert len(self.switch_points) == len(self.switch_taskid)

        # RNN model
        self.input_size = 33
        self.hidden_size = 400
        self.output_size = 17
        self.lr = 1e-4

        # test & plot
        self.test_every_trials = 500
        self.test_num_trials = 30
        self.plot_every_trials = 4000
        self.args= args

        self.exp_signature = self.exp_name +f'_{self.args}_'+\
        f'{"same_rnn" if self.same_rnn else "separate"}_{"gates" if self.use_gates else "nogates"}'+\
        f'_{"tc" if self.train_to_criterion else "nc"}'

        # Add tasks gradually with rehearsal 1 2 1 2 3 1 2 3 4 ...
        task_sub_seqs = [[(i, self.task_seq[i]) for i in range(s)] for s in range(2, len(self.task_seq))] # interleave tasks and add one task at a time
        self.task_seq_with_rehearsal = []
        for sub_seq in task_sub_seqs: self.task_seq_with_rehearsal+=sub_seq

