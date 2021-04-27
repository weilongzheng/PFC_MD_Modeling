import numpy as np

# Wei-Long version
class RikhyeTask():
    def __init__(self, Ntrain, Nextra, Ncontexts, inpsPerConext, blockTrain):
        self.Ncontexts = Ncontexts
        self.blockTrain = blockTrain
        self.inpsPerConext = inpsPerConext
        self.tsteps = 200
        self.cuesteps = 100 #200 # if 200, no delay period
        self.Ncues = self.Ncontexts * self.inpsPerConext

        if self.blockTrain:
            self.Nextra = Nextra            # add cycles to show if block1
            # learning is remembered
            Ntrain = Ntrain*self.Ncontexts + self.Nextra
        else:
            Ntrain *= self.Ncontexts

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
            n_switch = (self.Ntrain - self.Nextra) // self.Ncontexts
            taski = self.traini // (n_switch)
            # last block is just the first context again
            if self.traini >= self.Ntrain - self.Nextra:
                taski = 0
            cueList = self.get_cue_list(taski)
        else:
            cueList = self.get_cue_list()
        cues_order = self.get_cues_order(cueList)
        num_trial = cues_order.shape[0]

        inputs = np.zeros((self.tsteps * num_trial, self.Ncues))
        targets = np.zeros((self.tsteps * num_trial, 2))
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
            cueList = np.dstack(( np.repeat(taski,self.inpsPerConext),
                                    np.arange(self.inpsPerConext) ))
        else:
            # every possible (taski,cuei) combination
            cueList = np.dstack(( np.repeat(np.arange(self.Ncontexts),self.inpsPerConext),
                                    np.tile(np.arange(self.inpsPerConext),self.Ncontexts) ))
        return cueList[0]

class RikhyeTaskMultiCues():
    def __init__(self, Ntrain, Nextra, Ncontexts, inpsPerConext, blockTrain):
        self.Ncontexts = Ncontexts
        self.blockTrain = blockTrain
        self.inpsPerConext = inpsPerConext
        self.tsteps = 200
        self.cuesteps = 100 #200 # if 200, no delay period
        self.Ncues = self.Ncontexts * self.inpsPerConext
        self.tsteps_noise = 50

        if self.blockTrain:
            self.Nextra = Nextra            # add cycles to show if block1
            # learning is remembered
            Ntrain = Ntrain*self.Ncontexts + self.Nextra
        else:
            Ntrain *= self.Ncontexts

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
            n_switch = (self.Ntrain - self.Nextra) // self.Ncontexts
            taski = self.traini // (n_switch)
            # last block is just the first context again
            if self.traini >= self.Ntrain - self.Nextra:
                taski = 0
            cueList = self.get_cue_list(taski)
        else:
            cueList = self.get_cue_list()
        cues_order = self.get_cues_order(cueList)
        num_trial = cues_order.shape[0]

        inputs = np.zeros((self.tsteps * num_trial, self.Ncues))
        targets = np.zeros((self.tsteps * num_trial, 2))
        t_start = 0
        for taski, cuei in cues_order:
            cue, target = \
                self.get_cue_target(taski, cuei)
            inputs[t_start:t_start+self.cuesteps] = cue
            inputs[t_start+self.tsteps-self.tsteps_noise:t_start+self.tsteps] = np.ones_like(cue)
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
            cueList = np.dstack(( np.repeat(taski,self.inpsPerConext),
                                    np.arange(self.inpsPerConext) ))
        else:
            # every possible (taski,cuei) combination
            cueList = np.dstack(( np.repeat(np.arange(self.Ncontexts),self.inpsPerConext),
                                    np.tile(np.arange(self.inpsPerConext),self.Ncontexts) ))
        return cueList[0]
    
# Zhongxuan version
class RikhyeTaskBatch():
    def __init__(self, num_cueingcontext, num_cue, num_rule, rule, blocklen, block_cueingcontext, tsteps, cuesteps, batch_size):
        '''Generate Rikhye task dataset
        Parameters:
            num_cueingcontext: int, number of cueing contexts
            num_cue: int, number of cues in each cueing contexts
            num_rule: int, number of rules (e.g. attend to audition is a rule)
            rule: list of int, rule corresponding to one cue in one cueing context
            blocklen: list of int, trainlen of each block
            block_cueingcontext: list of int, cueing context trained in each block
            tsteps: int, length of a trial, equals to cuesteps + delaysteps
            cuesteps: int, length of showing cues
            batch_size: int
        '''
        self.num_cueingcontext = num_cueingcontext
        self.num_cue = num_cue
        self.num_rule = num_rule
        self.rule = rule

        self.blockrange = np.zeros_like(blocklen) # index range of each block
        for i in range(len(blocklen)):
            self.blockrange[i] = sum(blocklen[:i+1])

        self.block_cueingcontext = block_cueingcontext
        self.tsteps = tsteps
        self.cuesteps = cuesteps
        self.batch_size = batch_size

        # Initialize counter
        self.traini = 0

    def __call__(self, *args, **kwargs):
        """Return the input stimulus and target output for one cycle.
        Parameters:
            No parameter

        Returns:
            input: (n_time, n_input)
            target: (n_time, n_output)
        """

        inputs = np.zeros((self.tsteps * self.num_cue, self.batch_size, self.num_cue*self.num_cueingcontext))
        targets = np.zeros((self.tsteps * self.num_cue, self.batch_size, self.num_rule))

        for i in range(self.batch_size):
            blocki = np.argwhere(self.traini / self.blockrange < 1.0)
            if len(blocki) == 0:
                raise ValueError("the end")
            blocki = int(blocki[0])
            cueingcontext = self.block_cueingcontext[blocki]

            cueList = self.get_cue_list(cueingcontext)

            cues_order = np.random.permutation(cueList)


            t_start = 0
            for cueingcontext, cuei in cues_order:
                cue, target = self.get_cue_target(cueingcontext, cuei)
                inputs[t_start:t_start+self.cuesteps, i, :] = cue
                targets[t_start:t_start+self.tsteps, i, :] = target
                t_start += self.tsteps

            self.traini += 1

        return inputs, targets

    # Helper functions
    def get_cue_list(self, cueingcontext):
        '''
        Return:
        (cueingcontext, cuei) combinations for one training step
        '''
        cueList = np.dstack((np.repeat(cueingcontext,self.num_cue), np.arange(self.num_cue)))

        return cueList[0]

    def get_cue_target(self, cueingcontext, cuei):
        cue = np.zeros(self.num_cue*self.num_cueingcontext)
        cue[cueingcontext*self.num_cue + cuei] = 1.

        target = np.zeros(self.num_rule)
        target[self.rule[cueingcontext*self.num_cue + cuei]] = 1

        return cue, target