class BaseLogger(object):
    def __init__(self, config):
        self.losses = []
        self.stamps = []
        self.fix_perfs = []
        self.act_perfs = []
        for _ in range(config.num_task):
            self.fix_perfs.append([])
            self.act_perfs.append([])
        self.PFCouts_all = []

class PFCMDLogger(BaseLogger):
    def __init__(self, config):
        super(PFCMDLogger, self).__init__(config)
        self.MDouts_all = []
        self.MDpreTraces_all = []
        self.MDpreTraces_binary_all = []
        self.MDpreTrace_threshold_all = []
        self.MDpreTrace_binary_threshold_all = []
        self.wPFC2MD_list = []
        self.wMD2PFC_list = []