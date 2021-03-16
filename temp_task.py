
class RihkyeTask():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """Return the input stimulus and target output for one trial.

        Returns:
            input: (n_time, n_input)
            target: (n_time, n_output)
        """
        # if blockTrain,
        #  first half of trials is context1, second half is context2
        if self.blockTrain:
            taski = traini // ((Ntrain - Nextra) // self.Ntasks)
            # last block is just the first context again
            if traini >= Ntrain - Nextra: taski = 0
            cueList = self.get_cue_list(taski)
        else:
            cueList = self.get_cue_list()
        cues_order = self.get_cues_order(cueList)
        num_trial = cues_order.shape[0]
        for taski, cuei in cues_order:
            cue, target = \
                self.get_cue_target(taski, cuei)

        return cue, target

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

        if self.xorTask:
            if cuei in (0, 1):
                target = np.array((1., 0.))
            else:
                target = np.array((0., 1.))
        else:
            if cuei == 0:
                target = np.array((1., 0.))
            else:
                target = np.array((0., 1.))
        return cue, target