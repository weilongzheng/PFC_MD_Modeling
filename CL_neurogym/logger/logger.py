class BaseLogger(object):
    def __init__(self, config):
        self.losses = []
        self.stamps = []
        self.fix_perfs = []
        self.act_perfs = []
        self.md_activity = []
        for _ in range(config.num_task):
            self.fix_perfs.append([])
            self.act_perfs.append([])
            self.md_activity.append([])
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


class SerialLogger(object):
    def __init__(self, config):
        self.losses = []
        self.stamps = []
        self.accuracies = []
        self.fix_perfs = []
        self.act_perfs = []
        self.switch_trialxxbatch = []
        self.switch_task_id = []
        self.gradients = []
        
        for _ in range(len(config.tasks)):
            self.fix_perfs.append([])
            self.act_perfs.append([])
        self.PFCouts_all = []
        if config.save_detailed:
            self.rnn_activities = []
            self.inputs = []
            self.outputs = []
            self.labels = []

    def write_basic(self, stamp, loss, acc):
        self.stamps.append(stamp)
        self.losses.append(loss)
        self.accuracies.append(acc)
    def write_detailed(self, rnn_activities, inputs, outputs, labels,):#  gradients,):
        self.rnn_activities.append(rnn_activities)
        self.inputs.append(inputs)
        self.outputs.append(outputs)
        self.labels.append(labels)
        # self.gradients.append(gradients)