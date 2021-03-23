
import numpy as np


class RihkyeTask():
    def __init__(self, Ntrain, Ntasks, blockTrain):
        self.Ntasks = Ntasks
        self.blockTrain = blockTrain
        self.inpsPerTask = 2
        self.tsteps = 200
        self.cuesteps = 100
        self.Ncues = self.Ntasks * self.inpsPerTask

        if self.blockTrain:
            self.Nextra = 200            # add cycles to show if block1
            # learning is remembered
            Ntrain = Ntrain*self.Ntasks + self.Nextra
        else:
            Ntrain *= self.Ntasks

        self.Ntrain = Ntrain

        # Initialize counter
        self.traini = 0

    def __call__(self, *args, **kwargs):
        """Return the input stimulus and target output for one cycle.

        Returns:
            input: (n_time, n_input)
            target: (n_time, n_output)
        """
        # if blockTrain,
        #  first half of trials is context1, second half is context2
        if self.blockTrain:
            n_switch = (self.Ntrain - self.Nextra) // self.Ntasks
            taski = self.traini // (n_switch)
            # last block is just the first context again
            if self.traini >= self.Ntrain - self.Nextra:
                taski = 0
            cueList = self.get_cue_list(taski)
        else:
            cueList = self.get_cue_list()
        cues_order = self.get_cues_order(cueList)
        num_trial = cues_order.shape[0]

        inputs = np.zeros((self.tsteps * 2, 4))
        targets = np.zeros((self.tsteps * 2, 2))
        t_start = 0
        for taski, cuei in cues_order:
            cue, target = \
                self.get_cue_target(taski, cuei)
            inputs[t_start:t_start+self.cuesteps] = cue
            targets[t_start:t_start+self.tsteps] = target
            t_start += self.tsteps

        self.traini += 1

        return inputs, targets

    def get_cues_order(self, cues):
        cues_order = np.random.permutation(cues)
        return cues_order

    def get_cue_target(self, taski, cuei):
        cue = np.zeros(self.Ncues)
        inpBase = taski * 2
        if cuei in (0, 1):
            cue[inpBase + cuei] = 1.
        elif cuei == 3:
            cue[inpBase:inpBase + 2] = 1

        if cuei == 0:
            target = np.array((1., 0.))
        else:
            target = np.array((0., 1.))
        return cue, target

    def get_cue_list(self,taski=None):
        if taski is not None:
            # (taski,cuei) combinations for one given taski
            cueList = np.dstack(( np.repeat(taski,self.inpsPerTask),
                                    np.arange(self.inpsPerTask) ))
        else:
            # every possible (taski,cuei) combination
            cueList = np.dstack(( np.repeat(np.arange(self.Ntasks),self.inpsPerTask),
                                    np.tile(np.arange(self.inpsPerTask),self.Ntasks) ))
        return cueList[0]