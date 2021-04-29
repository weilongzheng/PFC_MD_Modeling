# -*- coding: utf-8 -*-

"""Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import numpy as np
import matplotlib.pyplot as plt
import sys, shelve

try:
    import torch
except ImportError:
    print('Torch not available')


class PFCMD():
    def __init__(self, PFC_G, PFC_G_off, learning_rate,
                 noiseSD, tauError, plotFigs=True, saveData=False):
        self.RNGSEED = 5  # random seed: 5
        np.random.seed([self.RNGSEED])

        self.Nsub = 200  # number of neurons per cue
        self.Ntasks = 2  # number of contexts = number of MD cells.
        self.xorTask = False  # use xor Task or simple 1:1 map task
        # self.xorTask = True                 # use xor Task or simple 1:1 map task

        if self.xorTask:
            self.inpsPerTask = 4  # number of cue combinations per task
        else:
            self.inpsPerTask = 2
        self.Ncues = self.Ntasks * self.inpsPerTask  # number of input cues
        self.Nneur = self.Nsub * (self.Ncues + 1)  # number of neurons
        self.Nout = 2  # number of outputs
        self.tau = 0.02
        self.dt = 0.001
        self.tsteps = 200  # number of timesteps in a trial
        self.cuesteps = 100  # number of time steps for which cue is on
        self.noiseSD = noiseSD
        self.saveData = saveData

        self.tau_times = 4  # 4
        self.Hebb_learning_rate = 1e-4  # 1e-4
        self.Num_MD = 6
        self.learning_rate = learning_rate  # if the learning rate is too large, 
        # output weights can change too much within a trial / training cycle,  
        # then the output interference depends on the order of cues within a cycle
        # typical value is 1e-5, and can vary from 1e-4 to 1e-6
        self.tauError = tauError  # smooth the error a bit, so that weights don't fluctuate

        self.MDeffect = True  # True                # whether to have MD present or not
        self.MDEffectType = 'submult'  # MD subtracts from across tasks and multiplies within task
        # self.MDEffectType = 'subadd'        # MD subtracts from across tasks and adds within task
        # self.MDEffectType = 'divadd'        # MD divides from across tasks and adds within task
        # self.MDEffectType = 'divmult'       # MD divides from across tasks and multiplies within task

        self.dirConn = False  # direct connections from cue to output, also learnable
        self.outExternal = True  # True: output neurons are external to the PFC
        #  (i.e. weights to and from (outFB) are not MD modulated)
        # False: last self.Nout neurons of PFC are output neurons
        self.outFB = False  # if outExternal, then whether feedback from output to reservoir
        self.noisePresent = False  # False           # add noise to all reservoir units

        self.positiveRates = True  # whether to clip rates to be only positive, G must also change

        self.MDlearn = True  # False                # whether MD should learn
        # TODO: possibly to make task representations disjoint (not just orthogonal)

        # self.MDstrength = None              # if None, use wPFC2MD, if not None as below, just use context directly
        # self.MDstrength = 0.                # a parameter that controls how much the MD disjoints task representations.
        self.MDstrength = 1.  # a parameter that controls how much the MD disjoints task representations.
        #  zero would be a pure reservoir, 1 would be full MDeffect
        # -1 for zero recurrent weights
        self.wInSpread = False  # Spread wIn also into other cue neurons to see if MD disjoints representations
        self.blockTrain = True  # first half of training is context1, second half is context2

        self.depress = False  # a depressive term if there is pre-post firing
        self.multiAttractorReservoir = False  # increase the reservoir weights within each cue
        #  all uniformly (could also try Hopfield style for the cue pattern)

        # Perhaps I shouldn't have self connections / autapses?!
        # Perhaps I should have sparse connectivity?

        if self.MDstrength < 0.: self.Jrec *= 0.

        # I don't want to have an if inside activation
        #  as it is called at each time step of the simulation
        # But just defining within __init__
        #  doesn't make it a member method of the class,
        #  hence the special self.__class__. assignment

        # wIn = np.random.uniform(-1,1,size=(self.Nneur,self.Ncues))


        # wDir and wOut are set in the main training loop
        if self.outExternal and self.outFB:
            self.wFB = np.random.uniform(-1, 1, size=(self.Nneur, self.Nout)) \
                       * self.G / np.sqrt(self.Nsub * 2) * PFC_G

        self.cue_eigvecs = np.zeros((self.Ncues, self.Nneur))
        self.plotFigs = plotFigs
        self.cuePlot = (0, 0)

        if self.saveData:
            self.fileDict = shelve.open('dataPFCMD/data_reservoir_PFC_MD' + \
                                        str(self.MDstrength) + \
                                        '_R' + str(self.RNGSEED) + \
                                        (
                                            '_xor' if self.xorTask else '') + '.shelve')

        self.meanAct = np.zeros(shape=(self.Ntasks * self.inpsPerTask, \
                                       self.tsteps, self.Nneur))

class PFC():
    def __init__(self, n_neuron, n_neuron_per_cue, positiveRates=True, MDeffect=True):
        self.Nneur = n_neuron
        self.Nsub = n_neuron_per_cue
        self.useMult = True
        self.noisePresent = False
        self.noiseSD = 1e-3#1e-3
        self.tau = 0.02
        self.dt = 0.001
                    
        self.positiveRates = positiveRates
        if self.positiveRates:
            # only +ve rates
            self.activation = lambda inp: np.clip(np.tanh(inp), 0, None)
        else:
            # both +ve/-ve rates as in Miconi
            self.activation = lambda inp: np.tanh(inp)

        self.G = 0.75  # determines also the cross-task recurrence
        # With MDeffect = True and MDstrength = 0, i.e. MD inactivated
        #  PFC recurrence is (1+PFC_G_off)*Gbase = (1+1.5)*0.75 = 1.875
        # So with MDeffect = False, ensure the same PFC recurrence for the pure reservoir
        if not MDeffect: self.G = 1.875

        self.init_activity()
        self.init_weights()

    def init_activity(self):
        self.xinp = np.random.uniform(0, 0.1, size=(self.Nneur))
        self.activity = self.activation(self.xinp)

    def init_weights(self):
        self.Jrec = np.random.normal(size=(self.Nneur, self.Nneur)) \
                    * self.G / np.sqrt(self.Nsub * 2)
        # make mean input to each row zero,
        #  helps to avoid saturation (both sides) for positive-only rates.
        #  see Nicola & Clopath 2016
        self.Jrec -= np.mean(self.Jrec, axis=1)[:, np.newaxis]
        # mean of rows i.e. across columns (axis 1),
        #  then expand with np.newaxis
        #   so that numpy's broadcast works on rows not columns

    def __call__(self, input, input_x=None, *args, **kwargs):
        """Run the network one step

        For now, consider this network receiving input from PFC,
        input stands for activity of PFC neurons
        output stands for output current to PFC neurons

        Args:
            input: array (n_neuron,)
            input_x: array (n_neuron,), modulatory input that multiplicatively
                interact with the neurons

        Returns:
            output: array (n_output,)
        """

        if input_x is None:
            input_x = np.zeros_like(input)
            
        xadd = np.dot(self.Jrec, self.activity)
        xadd += input_x + input # MD inputs
        
        self.xinp += self.dt / self.tau * (-self.xinp + xadd)

        if self.noisePresent:
            self.xinp += np.random.normal(size=(self.Nneur)) * self.noiseSD \
                    * np.sqrt(self.dt) / self.tau

        rout = self.activation(self.xinp)
        self.activity = rout
        return rout

    def update_weights(self, input, activity, output):
        self.trace = self.trace + activity
        w_input = self.w_input + input * self.trace
        w_output = self.w_output + input * self.trace


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

class OutputLayer():
    def __init__(self, n_input, n_out, dt):
        self.dt = dt
        self.tau = 0.02
        self.tauError = 0.001
        self.Nout = n_out
        self.Nneur = n_input
        self.learning_rate = 5e-6
        self.wOut = np.random.uniform(-1, 1,
                                      size=(
                                      self.Nout, self.Nneur)) / self.Nneur
        self.state = np.zeros(shape=self.Nout)
        self.error_smooth = np.zeros(shape=self.Nout)
        self.activation = lambda inp: np.clip(np.tanh(inp), 0, None)

    def __call__(self, input, target, *args, **kwargs):
        outAdd = np.dot(self.wOut, input)
        self.state += self.dt / self.tau * (-self.state + outAdd)
        output = self.activation(self.state)
        self.update_weights(input, output, target)
        return output

    def update_weights(self, input, output, target):
        """error-driven i.e. error*pre (perceptron like) learning"""
        error = output - target
        self.error_smooth += self.dt / self.tauError * (-self.error_smooth +
                                                        error)
        self.wOut += -self.learning_rate \
                     * np.outer(self.error_smooth, input)


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

    
    def shift(self, shift=1):
        '''
        shift Win to test shift problem in PFC_MD model
        '''
        self.wIn = np.roll(self.wIn, shift=shift, axis=0)


class FullNetwork():
    def __init__(self, Num_PFC, n_neuron_per_cue, Num_MD, num_active,
                 MDeffect=True):
        dt = 0.001
        self.pfc = PFC(Num_PFC, n_neuron_per_cue, MDeffect=MDeffect)
        self.sensory2pfc = SensoryInputLayer(
            n_sub=n_neuron_per_cue,
            n_cues=4,
            n_output=Num_PFC)
        self.pfc2out = OutputLayer(n_input=Num_PFC, n_out=2, dt=dt)
        self.pfc_output_t = np.array([])
        
        self.MDeffect = MDeffect
        if self.MDeffect:
            self.md = MD(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active,
                         dt=dt)
            self.md_output = np.zeros(Num_MD)
            index = np.random.permutation(Num_MD)
            self.md_output[index[:num_active]] = 1 # randomly set part of md_output to 1
            self.md_output_t = np.array([])
            #import pdb;pdb.set_trace()

    def __call__(self, input, target, *args, **kwargs):
        """
        Args:
             input: (n_time, n_input)
             target: (n_time, n_output)
             
        """
        self._check_shape(input, target)
        n_time = input.shape[0]
        tsteps = 200

        self.pfc.init_activity()  # Reinit PFC activity
        pfc_output = self.pfc.activity
        if self.MDeffect:
            self.md.init_activity()  # Reinit MD activity

        output = np.zeros((n_time, target.shape[-1]))
        self.pfc_output_t *= 0
        if self.MDeffect:
            self.md_output_t *= 0

        for i in range(n_time):
            input_t = input[i]
            target_t = target[i]
            
            if i % tsteps == 0: # Reinit activity for each trial
                self.pfc.init_activity()  # Reinit PFC activity
                pfc_output = self.pfc.activity
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            input2pfc = self.sensory2pfc(input_t)
            #import pdb;pdb.set_trace() 
            if self.MDeffect:
                self.md_output = self.md(pfc_output)

                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.pfc.Jrec, self.pfc.activity)
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / 2))
                md2pfc = md2pfc_weights * rec_inp  
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /2), self.md_output) 
                pfc_output = self.pfc(input2pfc, md2pfc)
#                pfc_output = pfc_output.reshape((1,pfc_output.shape[0]))
#                md_output = self.md_output
#                md_output = md_output.reshape((1,md_output.shape[0]))
                if i==0:
                    self.pfc_output_t = pfc_output.reshape((1,pfc_output.shape[0]))
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    
                    self.pfc_output_t = np.concatenate((self.pfc_output_t, pfc_output.reshape((1,pfc_output.shape[0]))),axis=0)
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)
            else:
                pfc_output = self.pfc(input2pfc)
                if i==0:
                    self.pfc_output_t = pfc_output.reshape((1,pfc_output.shape[0]))
                else:
                    self.pfc_output_t = np.concatenate((self.pfc_output_t, pfc_output.reshape((1,pfc_output.shape[0]))),axis=0)
            output[i] = self.pfc2out(pfc_output, target_t)
            
#        for i in range(n_time):
#            input_t = input[i]
#            target_t = target[i]
#
#            input2pfc = self.sensory2pfc(input_t)
#            if self.MDeffect:
#                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
#                rec_inp = np.dot(self.pfc.Jrec, self.pfc.activity)
#                md2pfc = (self.md.MD2PFCMult / np.round(self.md.Num_MD / 2))
#                md2pfc = md2pfc * rec_inp  # minmax 5
#                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /2), self.md_output) 
#                pfc_output = self.pfc(input2pfc, md2pfc)
#                self.md_output = self.md(pfc_output)
#                if i==50:
#                    self.pfc_output_t = pfc_output
#            else:
#                pfc_output = self.pfc(input2pfc)
#            output[i] = self.pfc2out(pfc_output, target_t)

        return output

    def _check_shape(self, input, target):
        assert len(input.shape) == 2
        assert len(target.shape) == 2
        assert input.shape[0] == target.shape[0]


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


class PytorchMD(nn.Module):
    def __init__(self, Nneur, Num_MD, num_active=1, positiveRates=True, dt=0.001):
        super().__init__()
        self.Nneur = Nneur
        self.Num_MD = Num_MD
        self.positiveRates = positiveRates
        self.num_active = num_active  # num_active: num MD active per context

        self.tau = 0.02
        self.tau_times = 4
        self.dt = dt
        self.tsteps = 200
        self.Hebb_learning_rate = 1e-4
        # working!
        Gbase = 0.75  # determines also the cross-task recurrence

        self.wPFC2MD = torch.normal(mean=0, std=1 / np.sqrt(
            self.Num_MD * self.Nneur), size=(self.Num_MD, self.Nneur))
        self.wMD2PFC = torch.normal(mean=0, std=1 / np.sqrt(
            self.Num_MD * self.Nneur), size=(self.Nneur, self.Num_MD))
        self.wMD2PFCMult = torch.normal(mean=0, std=1 / np.sqrt(
            self.Num_MD * self.Nneur), size=(self.Nneur, self.Num_MD))
        self.MDpreTrace = torch.zeros((self.Nneur))
        self.MDpostTrace = torch.zeros((self.Num_MD))
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
        self.MDinp = torch.zeros(self.Num_MD)

    def forward(self, input, *args, **kwargs):
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
                          (-self.MDinp + torch.matmul(self.wPFC2MD, input))
        else:  # shift PFC rates, so that mean is non-zero to turn MD on
            self.MDinp += self.dt / self.tauMD * \
                          (-self.MDinp + torch.matmul(self.wPFC2MD, (input + 1. / 2)))

        # num_active = np.round(self.Num_MD / self.Ntasks)
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
        self.MDpreTrace_threshold = torch.mean(self.MDpreTrace)
        # self.MDpreTrace_threshold = np.mean(self.MDpreTrace[:self.Nsub * self.Ncues])  # first 800 cells are cue selective
        # MDoutTrace_threshold = np.mean(MDoutTrace) #median
        MDoutTrace_threshold = 0.5
        wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * torch.outer(MDoutTrace - MDoutTrace_threshold,
                                                                self.MDpreTrace - self.MDpreTrace_threshold)

        # Update and clip the weights
        self.wPFC2MD = torch.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        self.wMD2PFC = torch.clip(self.wMD2PFC + 0.1 * (wPFC2MDdelta.T), -10., 0.)
        self.wMD2PFCMult = torch.clip(self.wMD2PFCMult + 0.1 * (wPFC2MDdelta.T), 0., 7. / self.G)

    def winner_take_all(self, MDinp):
        '''Winner take all on the MD
        '''

        # Thresholding
        MDout = torch.zeros(self.Num_MD)
        MDinp_sorted, ind_MDinp_sorted = torch.sort(MDinp)
        # num_active = np.round(self.Num_MD / self.Ntasks)

        MDthreshold = torch.mean(MDinp_sorted[-int(self.num_active) * 2:])
        # MDthreshold  = np.mean(MDinp)
        index_pos = torch.where(MDinp >= MDthreshold)
        index_neg = torch.where(MDinp < MDthreshold)
        MDout[index_pos] = 1
        MDout[index_neg] = 0
        return MDout


# class TempNetwork():
#     def __init__(self, Num_PFC, n_neuron_per_cue, Num_MD, num_active,
#                  MDeffect=True):
#         dt = 0.001
#         self.pfc = PytorchPFC(Num_PFC, n_neuron_per_cue, MDeffect=MDeffect)
#         self.sensory2pfc = SensoryInputLayer(
#             n_sub=n_neuron_per_cue,
#             n_cues=4,
#             n_output=Num_PFC)
#         self.sensory2pfc.torch(use_torch=True)
#         self.pfc2out = OutputLayer(n_input=Num_PFC, n_out=2, dt=dt)
#         self.pfc_output_t = np.array([])

#         self.MDeffect = MDeffect
#         if self.MDeffect:
#             self.md = MD(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active,
#                          dt=dt)
#             self.md_output = np.zeros(Num_MD)
#             index = np.random.permutation(Num_MD)
#             self.md_output[index[:num_active]] = 1 # randomly set part of md_output to 1
#             self.md_output_t = np.array([])

#     def __call__(self, input, target, *args, **kwargs):
#         """
#         Args:
#              input: (n_time, n_input)
#              target: (n_time, n_output)

#         """
#         self._check_shape(input, target)
#         n_time = input.shape[0]
#         tsteps = 200

#         self.pfc.init_activity()  # Reinit PFC activity
#         pfc_output = self.pfc.activity.numpy()
#         if self.MDeffect:
#             self.md.init_activity()  # Reinit MD activity

#         output = np.zeros((n_time, target.shape[-1]))
#         self.pfc_output_t *= 0
#         if self.MDeffect:
#             self.md_output_t *= 0

#         for i in range(n_time):
#             input_t = input[i]
#             target_t = target[i]
            
#             if i % tsteps == 0: # Reinit activity for each trial
#                 self.pfc.init_activity()  # Reinit PFC activity
#                 pfc_output = self.pfc.activity.numpy()
#                 if self.MDeffect:
#                     self.md.init_activity()  # Reinit MD activity

#             input2pfc = self.sensory2pfc(input_t)
#             #import pdb;pdb.set_trace() 
#             if self.MDeffect:
#                 self.md_output = self.md(pfc_output)

#                 self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
#                 rec_inp = np.dot(self.pfc.Jrec, self.pfc.activity)
#                 md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / 2))
#                 md2pfc = md2pfc_weights * rec_inp  
#                 md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /2), self.md_output) 
#                 pfc_output = self.pfc(torch.from_numpy(input2pfc),
#                                       torch.from_numpy(md2pfc)).numpy()

#                 if i==0:
#                     self.pfc_output_t = pfc_output.reshape((1,pfc_output.shape[0]))
#                     self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
#                 else:
                    
#                     self.pfc_output_t = np.concatenate((self.pfc_output_t, pfc_output.reshape((1,pfc_output.shape[0]))),axis=0)
#                     self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)
#             else:
#                 pfc_output = self.pfc(torch.from_numpy(input2pfc)).numpy()
#                 if i==0:
#                     self.pfc_output_t = pfc_output.reshape((1,pfc_output.shape[0]))
#                 else:
#                     self.pfc_output_t = np.concatenate((self.pfc_output_t, pfc_output.reshape((1,pfc_output.shape[0]))),axis=0)
#             output[i] = self.pfc2out(pfc_output, target_t)

#         return output

#     def _check_shape(self, input, target):
#         assert len(input.shape) == 2
#         assert len(target.shape) == 2
#         assert input.shape[0] == target.shape[0]


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
        #self.pfc_output_t = np.array([])

        self.MDeffect = MDeffect
        if self.MDeffect:
            self.md = MD(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active,
                         dt=dt)
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
        self.pfc_outputs = torch.zeros((n_time, self.pfc.Nneur))
        if self.MDeffect:
            self.md_output_t *= 0

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
            #import pdb;pdb.set_trace() 
            if self.MDeffect:
                self.md_output = self.md(pfc_output.detach().numpy())

                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.pfc.Jrec.detach().numpy(), self.pfc.activity.detach().numpy())
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.num_output))
                md2pfc = md2pfc_weights * rec_inp  
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.num_output), self.md_output) 
                #pfc_output = self.pfc(torch.from_numpy(input2pfc), torch.from_numpy(md2pfc)).numpy()
                
                pfc_output = self.pfc(input2pfc,torch.from_numpy(md2pfc))
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
                
#                pfc_output = self.pfc(input2pfc, torch.from_numpy(md2pfc)).detach().numpy()
#                pfc_output_t = pfc_output.reshape((1, pfc_output.shape[0]))
#                self.pfc_outputs[i, :] = torch.from_numpy(pfc_output_t)

                if i==0:
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)
            else:
                pfc_output = self.pfc(input2pfc)
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
#                pfc_output = self.pfc(input2pfc).numpy()
#                pfc_output_t = pfc_output.reshape((1, pfc_output.shape[0]))
#                self.pfc_outputs[i, :] = torch.from_numpy(pfc_output_t)
            
        outputs = self.pfc2out(self.pfc_outputs)
        outputs = torch.tanh(outputs)
            
        return outputs

    def _check_shape(self, input, target):
        assert len(input.shape) == self.num_output
        assert len(target.shape) == 2
        assert input.shape[0] == target.shape[0]

# test if pfc md plasticity can track shift in inputs
class PytorchPFCMD_Shift(nn.Module):
    def __init__(self, Num_PFC, n_neuron_per_cue, Num_MD, num_active, num_output, shift_step, MDeffect=True, noisePresent = False, ):
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
        #self.pfc_output_t = np.array([])

        self.MDeffect = MDeffect
        if self.MDeffect:
            self.md = MD(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active,
                         dt=dt)
            self.md_output = np.zeros(Num_MD)
            index = np.random.permutation(Num_MD)
            self.md_output[index[:num_active]] = 1 # randomly set part of md_output to 1
            self.md_output_t = np.array([])
        

        self.num_output = num_output
        self.shift_step = shift_step
        self.shift_step_cum = 0 # cumulative shift steps

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
        self.pfc_outputs = torch.zeros((n_time, self.pfc.Nneur))
        if self.MDeffect:
            self.md_output_t *= 0

        for i in range(n_time):
            input_t = input[i]
            target_t = target[i]
            
            if i % tsteps == 0: # Reinit activity for each trial
                self.pfc.init_activity()  # Reinit PFC activity
                pfc_output = self.pfc.activity
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            input2pfc = self.sensory2pfc(input_t)
            input2pfc = np.roll(input2pfc, shift=self.shift_step_cum)
            # try learnable input weights
            # input2pfc = self.PytorchSensory2pfc(input_t)
            #import pdb;pdb.set_trace() 
            if self.MDeffect:
                self.md_output = self.md(pfc_output.detach().numpy())

                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.pfc.Jrec.detach().numpy(), self.pfc.activity.detach().numpy())
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.num_output))
                md2pfc = md2pfc_weights * rec_inp  
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.num_output), self.md_output) 
                #pfc_output = self.pfc(torch.from_numpy(input2pfc), torch.from_numpy(md2pfc)).numpy()
                
                pfc_output = self.pfc(input2pfc,torch.from_numpy(md2pfc))
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
                
#                pfc_output = self.pfc(input2pfc, torch.from_numpy(md2pfc)).detach().numpy()
#                pfc_output_t = pfc_output.reshape((1, pfc_output.shape[0]))
#                self.pfc_outputs[i, :] = torch.from_numpy(pfc_output_t)

                if i==0:
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)
            else:
                pfc_output = self.pfc(input2pfc)
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
#                pfc_output = self.pfc(input2pfc).numpy()
#                pfc_output_t = pfc_output.reshape((1, pfc_output.shape[0]))
#                self.pfc_outputs[i, :] = torch.from_numpy(pfc_output_t)
        self.shift_step_cum += self.shift_step   
        outputs = self.pfc2out(self.pfc_outputs)
        outputs = torch.tanh(outputs)
            
        return outputs

    def _check_shape(self, input, target):
        assert len(input.shape) == self.num_output
        assert len(target.shape) == 2
        assert input.shape[0] == target.shape[0]

class PytorchPFCMD2(nn.Module): # Same as PytorchPFCMD but using PytorchMD.
    def __init__(self, Num_PFC, n_neuron_per_cue, Num_MD, num_active, num_output, MDeffect=True):
        super().__init__()

        dt = 0.001

        self.sensory2pfc = SensoryInputLayer(
            n_sub=n_neuron_per_cue,
            n_cues=4,
            n_output=Num_PFC)
        self.sensory2pfc.torch(use_torch=True)
        # try learnable input weights
        # self.PytorchSensory2pfc = nn.Linear(4, Num_PFC)

        self.pfc = PytorchPFC(Num_PFC, n_neuron_per_cue, MDeffect=MDeffect)

        # self.pfc2out = OutputLayer(n_input=Num_PFC, n_out=2, dt=dt)
        self.pfc2out = nn.Linear(Num_PFC, num_output)
        # self.pfc_output_t = np.array([])

        self.MDeffect = MDeffect
        if self.MDeffect:
            # self.md = MD(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active,
            #              dt=dt)
            self.md = PytorchMD(Num_PFC, Num_MD, num_active, dt=dt)
            self.md_output = np.zeros(Num_MD)
            index = np.random.permutation(Num_MD)
            self.md_output[index[:num_active]] = 1  # randomly set part of md_output to 1
            self.md_output_t = np.array([])

        self.num_output = num_output

    def forward(self, input, target, *args, **kwargs):
        """
        Args:
             input: (n_time, n_input)
             target: (n_time, n_output)

        """
        # self._check_shape(input, target)
        n_time = input.shape[0]
        tsteps = 200

        self.pfc.init_activity()  # Reinit PFC activity
        pfc_output = self.pfc.activity
        if self.MDeffect:
            self.md.init_activity()  # Reinit MD activity

        # output = torch.zeros((n_time, target.shape[-1]))
        # self.pfc_output_t *= 0
        self.pfc_outputs = torch.zeros((n_time, self.pfc.Nneur))
        if self.MDeffect:
            self.md_output_t *= 0

        for i in range(n_time):
            input_t = input[i]
            target_t = target[i]

            if i % tsteps == 0:  # Reinit activity for each trial
                self.pfc.init_activity()  # Reinit PFC activity
                pfc_output = self.pfc.activity
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            input2pfc = self.sensory2pfc(input_t)
            # try learnable input weights
            # input2pfc = self.PytorchSensory2pfc(input_t)
            # import pdb;pdb.set_trace()
            if self.MDeffect:
                # self.md_output = self.md(pfc_output.detach().numpy())
                self.md_output = self.md(pfc_output)

                # self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                # rec_inp = np.dot(self.pfc.Jrec.detach().numpy(), self.pfc.activity.detach().numpy())
                # md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.num_output))
                # md2pfc = md2pfc_weights * rec_inp
                # md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.num_output), self.md_output)
                # #pfc_output = self.pfc(torch.from_numpy(input2pfc), torch.from_numpy(md2pfc)).numpy()
                #
                # pfc_output = self.pfc(input2pfc,torch.from_numpy(md2pfc))
                # pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                # self.pfc_outputs[i, :] = pfc_output_t

                self.md.MD2PFCMult = torch.matmul(self.md.wMD2PFCMult, self.md_output)
                rec_inp = torch.matmul(self.pfc.Jrec, self.pfc.activity)
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.num_output))
                md2pfc = md2pfc_weights * rec_inp
                md2pfc += torch.matmul(self.md.wMD2PFC / np.round(self.md.Num_MD / self.num_output), self.md_output)
                # pfc_output = self.pfc(torch.from_numpy(input2pfc), torch.from_numpy(md2pfc)).numpy()

                pfc_output = self.pfc(input2pfc, (md2pfc))
                pfc_output_t = pfc_output.view(1, pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t

                #                pfc_output = self.pfc(input2pfc, torch.from_numpy(md2pfc)).detach().numpy()
                #                pfc_output_t = pfc_output.reshape((1, pfc_output.shape[0]))
                #                self.pfc_outputs[i, :] = torch.from_numpy(pfc_output_t)

                if i == 0:
                    self.md_output_t = self.md_output.reshape((1, self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate(
                        (self.md_output_t, self.md_output.reshape((1, self.md_output.shape[0]))), axis=0)
            else:
                pfc_output = self.pfc(input2pfc)
                pfc_output_t = pfc_output.view(1, pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
                #                pfc_output = self.pfc(input2pfc).numpy()
                #                pfc_output_t = pfc_output.reshape((1, pfc_output.shape[0]))
                #                self.pfc_outputs[i, :] = torch.from_numpy(pfc_output_t)

        outputs = self.pfc2out(self.pfc_outputs)
        outputs = torch.tanh(outputs)

        return outputs

    def _check_shape(self, input, target):
        assert len(input.shape) == self.num_output
        assert len(target.shape) == 2
        assert input.shape[0] == target.shape[0]

#n_time = 200
#n_neuron = 1000
#n_neuron_per_cue = 200
#Num_MD = 20
#num_active = 10 # num MD active per context
#n_output = 2
#pfc_md = TempNetwork(n_neuron,n_neuron_per_cue,Num_MD,num_active)
#input = np.random.randn(n_time, 4)
#target = np.random.randn(n_time, n_output)
#output = pfc_md(input, target)
#print(output.shape)


class Elman(nn.Module):
    """Elman RNN that can take in MD inputs.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
    Inputs:
        input: (seq_len, batch, input_size), external input; seq_len is set to 1
        hidden: (batch, hidden_size), initial hidden activity;
        mdinput: (batch, hidden_size), MD input;

    Acknowlegement:
        based on Robert Yang's CTRNN class
    """


    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        if nonlinearity == 'relu':
            self.activation = torch.relu
        else:
            self.activation = torch.tanh

        # Sensory input -> RNN
        self.input2h = nn.Linear(input_size, hidden_size, bias=False)

        # RNN -> RNN
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Reset RNN weights
        '''
        #nn.init.eye_(self.h2h.weight)
        #self.h2h.weight.data *= 0.5

        #nn.init.normal_(self.h2h.weight, mean=0.0, std=1.0)

        k = (1./self.hidden_size)**0.5
        nn.init.uniform_(self.h2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        
    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden, mdinput):
        '''
        Recurrence helper function
        '''
        pre_activation = self.input2h(input) + self.h2h(hidden)

        if mdinput is None:
            mdinput = torch.zeros_like(pre_activation)
        else:
            pre_activation += mdinput
        
        h_new = self.activation(pre_activation)

        return h_new

    def forward(self, input, hidden=None, mdinput=None):
        '''
        Propogate input through the network
        '''
        if hidden is None:
            hidden = self.init_hidden(input)

        # output = []
        # steps = range(input.size(0))
        # for i in steps:
        #     hidden = self.recurrence(input[i], hidden, mdinput)
        #     output.append(hidden)
        # output = torch.stack(output, dim=0)

        # TODO: input.shape has to be [1, batch_size, input_size]
        hidden = self.recurrence(input[0], hidden, mdinput)
        
        return hidden


class Elman_MD(nn.Module):
    """Elman RNN with a MD layer
    Parameters:
    input_size: int, RNN input size
    hidden_size: int, RNN hidden size
    output_size: int, output layer size
    num_layers: int, number of RNN layers
    nonlinearity: str, 'tanh' or 'relu', nonlinearity in RNN layers
    Num_MD: int, number of neurons in MD layer
    num_active: int, number of active neurons in MD layer (refer to top K winner-take-all)
    tsteps: int, length of a trial, equals to cuesteps + delaysteps
    """


    def __init__(self, input_size, hidden_size, output_size, num_layers, nonlinearity, Num_MD, num_active, tsteps, MDeffect=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.tsteps = tsteps

        # PFC layer / Elman RNN layer
        self.rnn = Elman(input_size, hidden_size, nonlinearity)

        # MD layer
        self.MDeffect = MDeffect
        dt = 0.001 # Hard-coded for now
        if self.MDeffect:
            self.md = MD(Nneur=hidden_size, Num_MD=Num_MD, num_active=num_active, dt=dt)
            #  initialize md_output
            self.md_output = np.zeros(Num_MD)
            index = np.random.permutation(Num_MD)
            self.md_output[index[:num_active]] = 1 # randomly set part of md_output to 1
            self.md_output_t = np.array([])

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Track parameters
        self.parm = dict()
        for name, param in self.named_parameters():
            self.parm[name] = param

    def forward(self, input, target):
        '''
        Propogate input through the network
        '''
        n_time = input.shape[0]
        batch_size = input.shape[1]

        RNN_output = torch.zeros((n_time, batch_size, self.hidden_size))
        #RNN_hidden_t = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        RNN_hidden_t = torch.zeros((batch_size, self.hidden_size))

        for t in range(n_time):
            input_t = input[t, ...].unsqueeze(dim=0)
            target_t = target[t, ...].unsqueeze(dim=0)
            
            # Reinit MD activity for each trial
            if t % self.tsteps == 0: 
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            if self.MDeffect:
                # Generate MD activities
                self.md_output = self.md(RNN_hidden_t.detach().numpy()[0, :]) # batch size should be 1
                # Generate MD -> PFC inputs
                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.parm['rnn.h2h.weight'].detach().numpy(), RNN_hidden_t.detach().numpy()[0, :])  # PFC recurrent inputs # batch size should be 1
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.output_size))
                md2pfc = md2pfc_weights * rec_inp                                                                # MD inputs - multiplicative term
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.output_size), self.md_output)    # MD inputs - additive term
                md2pfc = torch.from_numpy(md2pfc).view_as(RNN_hidden_t)
                # Generate RNN activities
                #RNN_output[t, :, :], RNN_hidden_t = self.rnn(input_t, RNN_hidden_t, md2pfc)
                RNN_hidden_t = self.rnn(input_t, RNN_hidden_t, md2pfc)
                RNN_output[t, :, :] = RNN_hidden_t
                # Collect MD activities
                if t==0:
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)

            else:
                #RNN_output[t, :, :], RNN_hidden_t = self.rnn(input_t, RNN_hidden_t)
                RNN_hidden_t = self.rnn(input_t, RNN_hidden_t)
                RNN_output[t, :, :] = RNN_hidden_t

        model_out = self.fc(RNN_output)
        model_out = torch.tanh(model_out)

        return model_out


class LSTM_MD(nn.Module):
    """LSTM with a MD layer (this models works only when MDeffect = False)
    Parameters:
    input_size: int, LSTM input size
    hidden_size: int, LSTM hidden size
    output_size: int, output layer size
    num_layers: int, number of LSTM layers
    Num_MD: int, number of neurons in MD layer
    num_active: int, number of active neurons in MD layer (refer to top K winner-take-all)
    tsteps: int, length of a trial, equals to cuesteps + delaysteps
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, Num_MD, num_active, tsteps, MDeffect=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.tsteps = tsteps
        
        dt = 0.001 # Hard-coded for now

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        # MD layer
        self.MDeffect = MDeffect
        if self.MDeffect:
            self.md = MD(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active, dt=dt)
            #  initialize md_output
            self.md_output = np.zeros(Num_MD)
            index = np.random.permutation(Num_MD)
            self.md_output[index[:num_active]] = 1 # randomly set part of md_output to 1
            self.md_output_t = np.array([])

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, target):

        n_time = input.shape[0]
        batch_size = input.shape[1]

        LSTM_output = torch.zeros((n_time, batch_size, self.hidden_size))
        LSTM_hidden_t = torch.zeros((self.num_layers, batch_size, self.hidden_size))
        LSTM_cell_t = torch.zeros((self.num_layers, batch_size, self.hidden_size))

        for t in range(n_time):
            input_t = input[t, ...].unsqueeze(dim=0)
            target_t = target[t, ...].unsqueeze(dim=0)
            
            # Reinit MD activity for each trial
            if t % self.tsteps == 0: 
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            # TODO: integrate MD into LSTM_MD
            if self.MDeffect:
                self.md_output = self.md(LSTM_hidden_t.detach().numpy()[:, 0, :]) # batch size should be 1

                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.pfc.Jrec.detach().numpy(), self.pfc.activity.detach().numpy())
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.num_output))
                md2pfc = md2pfc_weights * rec_inp  
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.num_output), self.md_output) 
                
                LSTM_hidden_t = self.pfc(input2pfc,torch.from_numpy(md2pfc))
                pfc_output_t = LSTM_hidden_t.view(1,LSTM_hidden_t.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t

                if i==0:
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)

            else:
                LSTM_output[t, :, :], (LSTM_hidden_t, LSTM_cell_t) = self.lstm(input_t, (LSTM_hidden_t, LSTM_cell_t))

        model_out = self.fc(LSTM_output)

        return model_out