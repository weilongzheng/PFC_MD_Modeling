class BaseLogger(object):
    def __init__(self):
        self.losses = []
        self.stamps = []
        self.fix_perfs = [[], []]
        self.act_perfs = [[], []]
        self.PFCouts_all = []

class PFCMDLogger(BaseLogger):
    def __init__(self):
        super(PFCMDLogger, self).__init__()
        self.MDouts_all = []
        self.MDpreTraces_all = []
        self.MDpreTraces_binary_all = []
        self.MDpreTrace_threshold_all = []
        self.MDpreTrace_binary_threshold_all = []
        self.wPFC2MD_list = []
        self.wMD2PFC_list = []