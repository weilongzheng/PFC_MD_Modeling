# -*- coding: utf-8 -*-

"""Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import numpy as np
import matplotlib.pyplot as plt
import sys, shelve

try:
    import torch
except ImportError:
    print('Torch not available')


# class PFCMD():
#     def __init__(self, PFC_G, PFC_G_off, learning_rate,
#                  noiseSD, tauError, plotFigs=True, saveData=False):
#         self.RNGSEED = 5  # random seed: 5
#         np.random.seed([self.RNGSEED])

#         self.Nsub = 200  # number of neurons per cue
#         self.Ntasks = 2  # number of contexts = number of MD cells.
#         self.xorTask = False  # use xor Task or simple 1:1 map task
#         # self.xorTask = True                 # use xor Task or simple 1:1 map task

#         if self.xorTask:
#             self.inpsPerTask = 4  # number of cue combinations per task
#         else:
#             self.inpsPerTask = 2
#         self.Ncues = self.Ntasks * self.inpsPerTask  # number of input cues
#         self.Nneur = self.Nsub * (self.Ncues + 1)  # number of neurons
#         self.Nout = 2  # number of outputs
#         self.tau = 0.02
#         self.dt = 0.001
#         self.tsteps = 200  # number of timesteps in a trial
#         self.cuesteps = 100  # number of time steps for which cue is on
#         self.noiseSD = noiseSD
#         self.saveData = saveData

#         self.tau_times = 4  # 4
#         self.Hebb_learning_rate = 1e-4  # 1e-4
#         self.Num_MD = 6
#         self.learning_rate = learning_rate  # if the learning rate is too large, 
#         # output weights can change too much within a trial / training cycle,  
#         # then the output interference depends on the order of cues within a cycle
#         # typical value is 1e-5, and can vary from 1e-4 to 1e-6
#         self.tauError = tauError  # smooth the error a bit, so that weights don't fluctuate

#         self.MDeffect = True  # True                # whether to have MD present or not
#         self.MDEffectType = 'submult'  # MD subtracts from across tasks and multiplies within task
#         # self.MDEffectType = 'subadd'        # MD subtracts from across tasks and adds within task
#         # self.MDEffectType = 'divadd'        # MD divides from across tasks and adds within task
#         # self.MDEffectType = 'divmult'       # MD divides from across tasks and multiplies within task

#         self.dirConn = False  # direct connections from cue to output, also learnable
#         self.outExternal = True  # True: output neurons are external to the PFC
#         #  (i.e. weights to and from (outFB) are not MD modulated)
#         # False: last self.Nout neurons of PFC are output neurons
#         self.outFB = False  # if outExternal, then whether feedback from output to reservoir
#         self.noisePresent = False  # False           # add noise to all reservoir units

#         self.positiveRates = True  # whether to clip rates to be only positive, G must also change

#         self.MDlearn = True  # False                # whether MD should learn
#         # TODO: possibly to make task representations disjoint (not just orthogonal)

#         # self.MDstrength = None              # if None, use wPFC2MD, if not None as below, just use context directly
#         # self.MDstrength = 0.                # a parameter that controls how much the MD disjoints task representations.
#         self.MDstrength = 1.  # a parameter that controls how much the MD disjoints task representations.
#         #  zero would be a pure reservoir, 1 would be full MDeffect
#         # -1 for zero recurrent weights
#         self.wInSpread = False  # Spread wIn also into other cue neurons to see if MD disjoints representations
#         self.blockTrain = True  # first half of training is context1, second half is context2

#         self.depress = False  # a depressive term if there is pre-post firing
#         self.multiAttractorReservoir = False  # increase the reservoir weights within each cue
#         #  all uniformly (could also try Hopfield style for the cue pattern)

#         # Perhaps I shouldn't have self connections / autapses?!
#         # Perhaps I should have sparse connectivity?

#         if self.MDstrength < 0.: self.Jrec *= 0.

#         # I don't want to have an if inside activation
#         #  as it is called at each time step of the simulation
#         # But just defining within __init__
#         #  doesn't make it a member method of the class,
#         #  hence the special self.__class__. assignment

#         # wIn = np.random.uniform(-1,1,size=(self.Nneur,self.Ncues))


#         # wDir and wOut are set in the main training loop
#         if self.outExternal and self.outFB:
#             self.wFB = np.random.uniform(-1, 1, size=(self.Nneur, self.Nout)) \
#                        * self.G / np.sqrt(self.Nsub * 2) * PFC_G

#         self.cue_eigvecs = np.zeros((self.Ncues, self.Nneur))
#         self.plotFigs = plotFigs
#         self.cuePlot = (0, 0)

#         if self.saveData:
#             self.fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD' + \
#                                         str(self.MDstrength) + \
#                                         '_R' + str(self.RNGSEED) + \
#                                         (
#                                             '_xor' if self.xorTask else '') + '.shelve')

#         self.meanAct = np.zeros(shape=(self.Ntasks * self.inpsPerTask, \
#                                        self.tsteps, self.Nneur))

# class PFC():
#     def __init__(self, n_neuron, n_neuron_per_cue, positiveRates=True, MDeffect=True):
#         self.Nneur = n_neuron
#         self.Nsub = n_neuron_per_cue
#         self.useMult = True
#         self.noisePresent = False
#         self.noiseSD = 1e-3#1e-3
#         self.tau = 0.02
#         self.dt = 0.001
                    
#         self.positiveRates = positiveRates
#         if self.positiveRates:
#             # only +ve rates
#             self.activation = lambda inp: np.clip(np.tanh(inp), 0, None)
#         else:
#             # both +ve/-ve rates as in Miconi
#             self.activation = lambda inp: np.tanh(inp)

#         self.G = 0.75  # determines also the cross-task recurrence
#         # With MDeffect = True and MDstrength = 0, i.e. MD inactivated
#         #  PFC recurrence is (1+PFC_G_off)*Gbase = (1+1.5)*0.75 = 1.875
#         # So with MDeffect = False, ensure the same PFC recurrence for the pure reservoir
#         if not MDeffect: self.G = 1.875

#         self.init_activity()
#         self.init_weights()

#     def init_activity(self):
#         self.xinp = np.random.uniform(0, 0.1, size=(self.Nneur))
#         self.activity = self.activation(self.xinp)

#     def init_weights(self):
#         self.Jrec = np.random.normal(size=(self.Nneur, self.Nneur)) \
#                     * self.G / np.sqrt(self.Nsub * 2)
#         # make mean input to each row zero,
#         #  helps to avoid saturation (both sides) for positive-only rates.
#         #  see Nicola & Clopath 2016
#         self.Jrec -= np.mean(self.Jrec, axis=1)[:, np.newaxis]
#         # mean of rows i.e. across columns (axis 1),
#         #  then expand with np.newaxis
#         #   so that numpy's broadcast works on rows not columns

#     def __call__(self, input, input_x=None, *args, **kwargs):
#         """Run the network one step

#         For now, consider this network receiving input from PFC,
#         input stands for activity of PFC neurons
#         output stands for output current to PFC neurons

#         Args:
#             input: array (n_neuron,)
#             input_x: array (n_neuron,), modulatory input that multiplicatively
#                 interact with the neurons

#         Returns:
#             output: array (n_output,)
#         """

#         if input_x is None:
#             input_x = np.zeros_like(input)
            
#         xadd = np.dot(self.Jrec, self.activity)
#         xadd += input_x + input # MD inputs
        
#         self.xinp += self.dt / self.tau * (-self.xinp + xadd)

#         if self.noisePresent:
#             self.xinp += np.random.normal(size=(self.Nneur)) * self.noiseSD \
#                     * np.sqrt(self.dt) / self.tau

#         rout = self.activation(self.xinp)
#         self.activity = rout
#         return rout

#     def update_weights(self, input, activity, output):
#         self.trace = self.trace + activity
#         w_input = self.w_input + input * self.trace
#         w_output = self.w_output + input * self.trace


class SensoryInputLayer():
    def __init__(self, n_sub, n_cues, n_output):
        # TODO: Hard-coded for now
        self.Ncues = n_cues
        self.Nsub = n_sub
        self.Nneur = n_output
        self.positiveRates = True

        self.wIn = np.zeros((self.Nneur, self.Ncues))
        self.cueFactor = 1.5
        if self.positiveRates:
            lowcue, highcue = 0.5, 1.
        else:
            lowcue, highcue = -1., 1
        for cuei in np.arange(self.Ncues):
            self.wIn[self.Nsub * cuei:self.Nsub * (cuei + 1), cuei] = \
                np.random.uniform(lowcue, highcue, size=self.Nsub) \
                * self.cueFactor
                
        # ramdom init input weights
        # self.wIn = np.random.uniform(0, 1, size=(self.Nneur, self.Ncues))
        
        # init input weights with Gaussian Distribution
#        self.wIn = np.zeros((self.Nneur, self.Ncues))
#        self.wIn = np.random.normal(0, 1, size=(self.Nneur, self.Ncues))
#        self.wIn[self.wIn<0] = 0
            
        self._use_torch = False

    def __call__(self, input):
        if self._use_torch:
            input = input.numpy()

        output = np.dot(self.wIn, input)

        if self._use_torch:
            #output = torch.from_numpy(output, dtype=torch.float).astype(torch.float)
            output = torch.from_numpy(output).type(torch.float)

        return output

    def torch(self, use_torch=True):
        self._use_torch = use_torch

    
    def shift(self, shift=0):
        '''
        shift Win to test shift problem in PFC_MD model
        '''
        self.wIn = np.roll(self.wIn, shift=shift, axis=0)


import torch
from torch import nn

class PytorchPFC(nn.Module):
    def __init__(self, n_neuron, n_neuron_per_cue, positiveRates=True, MDeffect=True, noisePresent = False):
        super().__init__()
        self.Nneur = n_neuron
        self.Nsub = n_neuron_per_cue
        self.useMult = True
        self.noisePresent = noisePresent
        self.noiseSD = 1e-2  # 1e-3
        self.tau = 0.02
        self.dt = 0.001

        self.positiveRates = positiveRates
        if self.positiveRates:
            # only +ve rates
            self.activation = lambda inp: torch.clip(torch.tanh(inp), 0, None)
        else:
            # both +ve/-ve rates as in Miconi
            self.activation = lambda inp: torch.tanh(inp)

        self.G = 0.75  # determines also the cross-task recurrence
        if not MDeffect: self.G = 1.875

        self.init_activity()
        self.init_weights()

    def init_activity(self):
        self.xinp = torch.rand(self.Nneur) * 0.1
        self.activity = self.activation(self.xinp)

    def init_weights(self):
        self.Jrec = torch.normal(mean=0, std=self.G / np.sqrt(self.Nsub * 2),
                                 size=(self.Nneur, self.Nneur))
        # make mean input to each row zero,
        #  helps to avoid saturation (both sides) for positive-only rates.
        #  see Nicola & Clopath 2016
        # mean of rows i.e. across columns (axis 1),
        #  then expand with np.newaxis
        #   so that numpy's broadcast works on rows not columns
        self.Jrec -= torch.mean(self.Jrec, dim=1).unsqueeze_(dim=1)
        self.Jrec.requires_grad = True # Block when using PytorchMD and PytorchPFCMD.

    def forward(self, input, input_x=None):
        """Run the network one step

        For now, consider this network receiving input from PFC,
        input stand for activity of PFC neurons
        output stand for output current to PFC neurons

        Args:
            input: array (n_neuron,)
            input_x: array (n_neuron,), modulatory input that multiplicatively
                interact with the neurons

        Returns:
            output: array (n_output,)
        """
        if input_x is None:
            input_x = torch.zeros(input.shape)

        xadd = torch.matmul(self.Jrec, self.activity)
        xadd += input_x + input  # MD inputs
        self.xinp += self.dt / self.tau * (-self.xinp + xadd)
        
        if self.noisePresent:
            self.xinp += torch.normal(mean=0, std=self.noiseSD * np.sqrt(self.dt) / self.tau, size=(self.Nneur,))
                    
        rout = self.activation(self.xinp)
        self.activity = rout
        return rout

#model = PytorchPFC(n_neuron=10, n_neuron_per_cue=1)
#input = torch.randn(10)
#output = model(input)
#print(output.shape)


class MD():
    def __init__(self, Nneur, Num_MD, num_active=1, positiveRates=True,
                 dt=0.001):
        self.Nneur = Nneur
        self.Num_MD = Num_MD
        self.positiveRates = positiveRates
        self.num_active = num_active # num_active: num MD active per context

        self.tau = 0.02
        self.tau_times = 4
        self.dt = dt
        self.tsteps = 200
        self.Hebb_learning_rate = 1e-4
        # working!
        Gbase = 0.75  # determines also the cross-task recurrence
#        self.MDstrength = 1
#        if self.MDstrength is None:
#            MDval = 1.
#        elif self.MDstrength < 0.:
#            MDval = 0.
#        else:
#            MDval = self.MDstrength
#        # subtract across tasks (task with higher MD suppresses cross-tasks)
#        self.wMD2PFC = np.ones(shape=(self.Nneur, self.Num_MD)) * (
#            -10.) * MDval
#        self.useMult = True
#        # multiply recurrence within task, no addition across tasks
#        ## choose below option for cross-recurrence
#        ##  if you want "MD inactivated" (low recurrence) state
#        ##  as the state before MD learning
#        # self.wMD2PFCMult = np.zeros(shape=(self.Nneur,self.Ntasks))
#        # choose below option for cross-recurrence
#        #  if you want "reservoir" (high recurrence) state
#        #  as the state before MD learning (makes learning more difficult)
#        self.wMD2PFCMult = np.ones(shape=(self.Nneur, self.Num_MD)) \
#                           * PFC_G_off / Gbase * (1 - MDval)
#        # threshold for sharp sigmoid (0.1 width) transition of MDinp
#        self.MDthreshold = 0.4
#
#        # With MDeffect = True and MDstrength = 0, i.e. MD inactivated
#        #  PFC recurrence is (1+PFC_G_off)*Gbase = (1+1.5)*0.75 = 1.875
#        # So with MDeffect = False, ensure the same PFC recurrence for the pure reservoir
#        # if not self.MDeffect: Gbase = 1.875

        self.wPFC2MD = np.random.normal(0, 1 / np.sqrt(
            self.Num_MD * self.Nneur), size=(self.Num_MD, self.Nneur))
        self.wMD2PFC = np.random.normal(0, 1 / np.sqrt(
            self.Num_MD * self.Nneur), size=(self.Nneur, self.Num_MD))
        self.wMD2PFCMult = np.random.normal(0, 1 / np.sqrt(
            self.Num_MD * self.Nneur), size=(self.Nneur, self.Num_MD))
        self.MDpreTrace = np.zeros(shape=(self.Nneur))
        self.MDpostTrace = np.zeros(shape=(self.Num_MD))
        self.MDpreTrace_threshold = 0

        # Choose G based on the type of activation function
        # unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if self.positiveRates:
            self.G = Gbase
            self.tauMD = self.tau * self.tau_times  ##self.tau
        else:
            self.G = Gbase
            self.MDthreshold = 0.4
            self.tauMD = self.tau * 10 * self.tau_times
        self.init_activity()
        
    def init_activity(self):
        self.MDinp = np.zeros(shape=self.Num_MD)
        
    def __call__(self, input, *args, **kwargs):
        """Run the network one step

        For now, consider this network receiving input from PFC,
        input stands for activity of PFC neurons
        output stands for output current to MD neurons

        Args:
            input: array (n_input,)
            

        Returns:
            output: array (n_output,)
        """
        # MD decays 10x slower than PFC neurons,
        #  so as to somewhat integrate PFC input
        if self.positiveRates:
            self.MDinp += self.dt / self.tauMD * \
                     (-self.MDinp + np.dot(self.wPFC2MD, input))
        else:  # shift PFC rates, so that mean is non-zero to turn MD on
            self.MDinp += self.dt / self.tauMD * \
                     (-self.MDinp + np.dot(self.wPFC2MD, (input + 1. / 2)))
                     
        #num_active = np.round(self.Num_MD / self.Ntasks)
        MDout = self.winner_take_all(self.MDinp)

        self.update_weights(input, MDout)

        return MDout

    def update_trace(self, rout, MDout):
        # MD presynaptic traces filtered over 10 trials
        # Ideally one should weight them with MD syn weights,
        #  but syn plasticity just uses pre!
        self.MDpreTrace += 1. / self.tsteps / 5. * \
                           (-self.MDpreTrace + rout)
        self.MDpostTrace += 1. / self.tsteps / 5. * \
                            (-self.MDpostTrace + MDout)
        # MDoutTrace =  self.MDpostTrace

        MDoutTrace = self.winner_take_all(self.MDpostTrace)
#        MDoutTrace = np.zeros(self.Num_MD)
#        MDpostTrace_sorted = np.sort(self.MDpostTrace)
#        num_active = np.round(self.Num_MD / self.Ntasks)
#        # MDthreshold  = np.mean(MDpostTrace_sorted[-4:])
#        MDthreshold = np.mean(
#            MDpostTrace_sorted[-int(num_active) * 2:])
#        # MDthreshold  = np.mean(self.MDpostTrace)
#        index_pos = np.where(self.MDpostTrace >= MDthreshold)
#        index_neg = np.where(self.MDpostTrace < MDthreshold)
#        MDoutTrace[index_pos] = 1
#        MDoutTrace[index_neg] = 0
        return MDoutTrace

    def update_weights(self, rout, MDout):
        """Update weights with plasticity rules.

        Args:
            rout: input to MD
            MDout: activity of MD
        """
        MDoutTrace = self.update_trace(rout, MDout)
        #                    if self.MDpostTrace[0] > self.MDpostTrace[1]: MDoutTrace = np.array([1,0])
        #                    else: MDoutTrace = np.array([0,1])
        self.MDpreTrace_threshold = np.mean(self.MDpreTrace)
        #self.MDpreTrace_threshold = np.mean(self.MDpreTrace[:self.Nsub * self.Ncues])  # first 800 cells are cue selective
        # MDoutTrace_threshold = np.mean(MDoutTrace) #median
        MDoutTrace_threshold = 0.5  
        wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)

        # Update and clip the weights
        self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        self.wMD2PFC = np.clip(self.wMD2PFC + 0.1 * (wPFC2MDdelta.T), -10., 0.)
        self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 0.1 * (wPFC2MDdelta.T), 0.,7. / self.G)

    def winner_take_all(self, MDinp):
        '''Winner take all on the MD
        '''

        # Thresholding
        MDout = np.zeros(self.Num_MD)
        MDinp_sorted = np.sort(MDinp)
        # num_active = np.round(self.Num_MD / self.Ntasks)

        MDthreshold = np.mean(MDinp_sorted[-int(self.num_active) * 2:])
        # MDthreshold  = np.mean(MDinp)
        index_pos = np.where(MDinp >= MDthreshold)
        index_neg = np.where(MDinp < MDthreshold)
        MDout[index_pos] = 1
        MDout[index_neg] = 0
        return MDout


# Zhongxuan: developing MD layer
class MD_dev():
    def __init__(self, Nneur, Num_MD, num_active=1, positiveRates=True,
                 dt=0.001):
        self.Nneur = Nneur # number of PFC neurons
        self.Num_MD = Num_MD # number of MD neurons
        self.positiveRates = positiveRates
        self.num_active = num_active # num_active: num MD active per context

        self.tau = 0.02
        self.tau_times = 4
        self.dt = dt
        self.tsteps = 200
        self.Hebb_learning_rate = 1e-4
        # working!
        Gbase = 0.75  # also determines the cross-task recurrence
#        self.MDstrength = 1
#        if self.MDstrength is None:
#            MDval = 1.
#        elif self.MDstrength < 0.:
#            MDval = 0.
#        else:
#            MDval = self.MDstrength
#        # subtract across tasks (task with higher MD suppresses cross-tasks)
#        self.wMD2PFC = np.ones(shape=(self.Nneur, self.Num_MD)) * (
#            -10.) * MDval
#        self.useMult = True
#        # multiply recurrence within task, no addition across tasks
#        ## choose below option for cross-recurrence
#        ##  if you want "MD inactivated" (low recurrence) state
#        ##  as the state before MD learning
#        # self.wMD2PFCMult = np.zeros(shape=(self.Nneur,self.Ntasks))
#        # choose below option for cross-recurrence
#        #  if you want "reservoir" (high recurrence) state
#        #  as the state before MD learning (makes learning more difficult)
#        self.wMD2PFCMult = np.ones(shape=(self.Nneur, self.Num_MD)) \
#                           * PFC_G_off / Gbase * (1 - MDval)
#        # threshold for sharp sigmoid (0.1 width) transition of MDinp
#        self.MDthreshold = 0.4
#
#        # With MDeffect = True and MDstrength = 0, i.e. MD inactivated
#        #  PFC recurrence is (1+PFC_G_off)*Gbase = (1+1.5)*0.75 = 1.875
#        # So with MDeffect = False, ensure the same PFC recurrence for the pure reservoir
#        # if not self.MDeffect: Gbase = 1.875

        # initialize PFC-MD weights
        self.wPFC2MD = np.random.normal(0, \
                                        1 / np.sqrt(self.Num_MD * self.Nneur), \
                                        size=(self.Num_MD, self.Nneur))
        self.wMD2PFC = np.random.normal(0, \
                                        1 / np.sqrt(self.Num_MD * self.Nneur), \
                                        size=(self.Nneur, self.Num_MD))
        self.wMD2PFCMult = np.random.normal(0, \
                                            1 / np.sqrt(self.Num_MD * self.Nneur), \
                                            size=(self.Nneur, self.Num_MD))
        
        # initialize MD traces & MD threshold
        self.MDpreTrace = np.zeros(shape=(self.Nneur))
        self.MDpostTrace = np.zeros(shape=(self.Num_MD))
        self.MDpreTrace_threshold = 0

        # Choose G based on the type of activation function
        # unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if self.positiveRates:
            self.G = Gbase
            self.tauMD = self.tau * self.tau_times  ##self.tau
        else:
            self.G = Gbase
            self.MDthreshold = 0.4
            self.tauMD = self.tau * 10 * self.tau_times
        self.init_activity()
        
    def init_activity(self):
        self.MDinp = np.zeros(shape=self.Num_MD)
        
    def __call__(self, input, *args, **kwargs):
        """Run the network one step

        For now, consider this network receiving input from PFC,
        input stands for activity of PFC neurons
        output stands for output current to MD neurons

        Args:
            input: array (n_input,)
            

        Returns:
            output: array (n_output,)
        """
        # MD decays 10x slower than PFC neurons,
        #  so as to somewhat integrate PFC input
        if self.positiveRates:
            self.MDinp += self.dt / self.tauMD * \
                         (-self.MDinp + np.dot(self.wPFC2MD, input))
            # self.MDinp = np.dot(self.wPFC2MD, input)
        else:  # shift PFC rates, so that mean is non-zero to turn MD on
            self.MDinp += self.dt / self.tauMD * \
                         (-self.MDinp + np.dot(self.wPFC2MD, (input + 1. / 2)))
                     
        # num_active = np.round(self.Num_MD / self.Ntasks)
        MDout = self.winner_take_all(self.MDinp)

        # self.update_weights(input, MDout)

        return MDout

    def update_trace(self, rout, MDout):
        # MD presynaptic traces filtered over 10 trials
        # Ideally one should weight them with MD syn weights,
        #  but syn plasticity just uses pre!

        # original self.MDpreTrace  += 1. / self.tsteps / 5.0 * (-self.MDpreTrace + rout)
        # self.MDpreTrace  += 1. / 5.0 * (-self.MDpreTrace + rout)
        self.MDpreTrace  = rout

        # original self.MDpostTrace += 1. / self.tsteps / 5.0 * (-self.MDpostTrace + MDout)
        # self.MDpostTrace += 1. / 5.0 * (-self.MDpostTrace + MDout)
        self.MDpostTrace = MDout
        
        MDoutTrace = self.winner_take_all(self.MDpostTrace)

        return MDoutTrace

    def winner_take_all(self, MDinp):
        '''Winner take all on the MD
        '''

        # Thresholding
        MDout = np.zeros(self.Num_MD)
        MDinp_sorted = np.sort(MDinp)
        # num_active = np.round(self.Num_MD / self.Ntasks)

        MDthreshold = np.mean(MDinp_sorted[-int(self.num_active) * 2:])
        # MDthreshold  = np.mean(MDinp)
        index_pos = np.where(MDinp >= MDthreshold)
        index_neg = np.where(MDinp < MDthreshold)
        # binary MD outputs
        MDout[index_pos] = 1
        MDout[index_neg] = 0
        return MDout

    def update_weights(self, rout, MDout):
        """Update weights with plasticity rules.

        Args:
            rout: input to MD
            MDout: activity of MD
        """
        MDoutTrace = self.update_trace(rout, MDout)

        # if self.MDpostTrace[0] > self.MDpostTrace[1]: MDoutTrace = np.array([1,0])
        # else: MDoutTrace = np.array([0,1])

         
        # original self.MDpreTrace_threshold = np.mean(self.MDpreTrace)
        # self.MDpreTrace_threshold = np.median(np.sort(self.MDpreTrace)[-800:])
        self.MDpreTrace_threshold = np.mean(self.MDpreTrace)
        
        # MDoutTrace_threshold = np.mean(MDoutTrace) #median
        MDoutTrace_threshold = 0.5 # original 0.5  

        
        # original wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)
        wPFC2MDdelta = 50.0 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)
        
        # Update and clip the weights
        # original
        # self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        # self.wMD2PFC = np.clip(self.wMD2PFC + 0.1 * (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 0.1 * (wPFC2MDdelta.T), 0.,7. / self.G)
        
        # keep wPFC2MD and wMD2PFC symmetrical
        # self.wPFC2MD = np.clip(self.wPFC2MD + 1.0 *  wPFC2MDdelta,      0., 10.)
        # self.wMD2PFC = np.clip(self.wMD2PFC + 1.0 * (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 1.0 * (wPFC2MDdelta.T), 0., 7. / self.G)

        # Increase the inhibition
        self.wPFC2MD = np.clip(self.wPFC2MD + 0.05 * wPFC2MDdelta, 0., 1.)
        self.wMD2PFC = np.clip(self.wMD2PFC + 1.0 * (wPFC2MDdelta.T), -10., 0.)
        self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 1.0 * (wPFC2MDdelta.T), 0.,7. / self.G)

    # should not shift PFC-MD weights in the shift problem
    def shift_weights(self, shift=0):
        self.wPFC2MD = np.roll(self.wPFC2MD, shift=shift, axis=1)
        self.wMD2PFC = np.roll(self.wMD2PFC, shift=shift, axis=0)


class PytorchPFCMD(nn.Module):
    def __init__(self, Num_PFC, n_neuron_per_cue, Num_MD, num_active, num_output, MDeffect=True, noisePresent = False):
        super().__init__()

        dt = 0.001

        self.sensory2pfc = SensoryInputLayer(
            n_sub=n_neuron_per_cue,
            n_cues=4,
            n_output=Num_PFC)
        self.sensory2pfc.torch(use_torch=True)
        # try learnable input weights
        # self.PytorchSensory2pfc = nn.Linear(4, Num_PFC)

        self.pfc = PytorchPFC(Num_PFC, n_neuron_per_cue, MDeffect=MDeffect, noisePresent = noisePresent)

        #self.pfc2out = OutputLayer(n_input=Num_PFC, n_out=2, dt=dt)
        self.pfc2out = nn.Linear(Num_PFC, num_output)


        self.MDeffect = MDeffect
        if self.MDeffect:
            # use MD_dev here
            self.md = MD_dev(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active, dt=dt)
            self.md_output = np.zeros(Num_MD)
            index = np.random.permutation(Num_MD)
            self.md_output[index[:num_active]] = 1 # randomly set part of md_output to 1
            self.md_output_t = np.array([])

        self.num_output = num_output

    def forward(self, input, target, *args, **kwargs):
        """
        Args:
             input: (n_time, n_input)
             target: (n_time, n_output)

        """
        #self._check_shape(input, target)
        n_time = input.shape[0]
        tsteps = 200

        self.pfc.init_activity()  # Reinit PFC activity
        pfc_output = self.pfc.activity
        if self.MDeffect:
            self.md.init_activity()  # Reinit MD activity

        #output = torch.zeros((n_time, target.shape[-1]))
        #self.pfc_output_t *= 0

        # initialize variables for saving important network activities
        self.pfc_outputs = torch.zeros((n_time, self.pfc.Nneur))
        self.md_preTraces = np.zeros(shape=(n_time, self.pfc.Nneur))
        self.md_preTrace_thresholds = np.zeros(shape=(n_time, 1))

        if self.MDeffect:
            self.md_output_t *= 0
            # reinitialize pretrace every cycle
            # self.md.MDpreTrace = np.zeros(shape=(self.pfc.Nneur))

        for i in range(n_time):
            input_t = input[i]
            target_t = target[i]
            
            if i % tsteps == 0: # Reinit activity for each trial
                self.pfc.init_activity()  # Reinit PFC activity
                pfc_output = self.pfc.activity
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            input2pfc = self.sensory2pfc(input_t)
            # try learnable input weights
            # input2pfc = self.PytorchSensory2pfc(input_t) 
            if self.MDeffect:
                self.md_output = self.md(pfc_output.detach().numpy())

                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.pfc.Jrec.detach().numpy(), self.pfc.activity.detach().numpy())
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.num_output))
                md2pfc = md2pfc_weights * rec_inp  
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.num_output), self.md_output)
                
                pfc_output = self.pfc(input2pfc,torch.from_numpy(md2pfc))

                # save important network activities
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
                self.md_preTraces[i, :] = self.md.MDpreTrace
                self.md_preTrace_thresholds[i, :] = self.md.MDpreTrace_threshold

                if i==0:
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)
            
            else:
                pfc_output = self.pfc(input2pfc)
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
        
        # update PFC-MD weights every training cycle
        if self.MDeffect:
            self.md.update_weights(np.mean(self.pfc_outputs.detach().numpy(), axis=0), np.mean(self.md_output_t, axis=0))
        
        outputs = self.pfc2out(self.pfc_outputs)
        outputs = torch.tanh(outputs)
            
        return outputs

    def _check_shape(self, input, target):
        assert len(input.shape) == self.num_output
        assert len(target.shape) == 2
        assert input.shape[0] == target.shape[0]