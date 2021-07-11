# -*- coding: utf-8 -*-

"""Some reservoir tweaks are inspired by Nicola and Clopath, arxiv, 2016 and Miconi 2016."""

import numpy as np
import matplotlib.pyplot as plt
import sys, shelve

try:
    import torch
    from torch import nn
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
    def __init__(self, Ntasks, input_size_per_task, n_output, n_sub):
        self.Ntasks = Ntasks
        self.input_size = Ntasks*input_size_per_task
        self.input_size_per_task = input_size_per_task

        self.Nsub = n_sub
        self.Nneur = n_output
        self.positiveRates = True
        self.wIn = np.zeros((self.Nneur, self.input_size))
        self.cueFactor = 1.5

        if self.positiveRates:
            lowcue, highcue = 0.5, 1.
        else:
            lowcue, highcue = -1., 1.
        for taski in range(self.Ntasks):
            self.wIn[self.Nsub*taski : self.Nsub*(taski+1),  \
                     self.input_size_per_task*taski :  self.input_size_per_task*(taski+1)] = \
                     np.random.uniform(lowcue, highcue, size=(self.Nsub, self.input_size_per_task)) * self.cueFactor

        # init input weights as in pytorch linear layer
        # k = np.sqrt(self.input_size_per_task)
        # self.wIn[self.Nsub*taski : self.Nsub*(taski+1),  \
        #          self.input_size_per_task*taski : self.input_size_per_task*(taski+1)] = \
        #          np.random.uniform(-k, k, size=(self.Nsub, self.input_size_per_task))

        # ramdom init input weights
        # self.wIn = np.random.uniform(0, 1, size=(self.Nneur, self.Ncues))
        
        # init input weights with Gaussian Distribution
        # self.wIn = np.zeros((self.Nneur, self.Ncues))
        # self.wIn = np.random.normal(0, 1, size=(self.Nneur, self.Ncues))
        # self.wIn[self.wIn<0] = 0
         
        self._use_torch = False

    def __call__(self, input):
        if self._use_torch:
            input = input.numpy()

        output = np.dot(self.wIn, input)

        if self._use_torch:
            output = torch.from_numpy(output).type(torch.float)

        return output

    def torch(self, use_torch=True):
        self._use_torch = use_torch

    
    def shift(self, shift=0):
        '''
        shift Win to test shift problem in PFC_MD model
        '''
        self.wIn = np.roll(self.wIn, shift=shift, axis=0)


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
        self.Jrec = torch.normal(mean=0,
                                 std=self.G / np.sqrt(self.Nsub * 2),
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
        xadd += input_x + input  # input_x: MD inputs
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
        Gbase = 0.75  # determines also the cross-task recurrence

        # initialize weights
        self.wPFC2MD = np.random.normal(0,
                                        1 / np.sqrt(self.Num_MD * self.Nneur),
                                        size=(self.Num_MD, self.Nneur))
        self.wMD2PFC = np.random.normal(0,
                                        1 / np.sqrt(self.Num_MD * self.Nneur),
                                        size=(self.Nneur, self.Num_MD))
        self.wMD2PFCMult = np.random.normal(0,
                                            1 / np.sqrt(self.Num_MD * self.Nneur),
                                            size=(self.Nneur, self.Num_MD))
        # initialize activities
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
                     
        MDout = self.winner_take_all(self.MDinp)

        self.update_weights(input, MDout)

        return MDout

    def update_trace(self, rout, MDout):
        # MD presynaptic traces filtered over 10 trials
        # Ideally one should weight them with MD syn weights,
        #  but syn plasticity just uses pre!

        self.MDpreTrace += 1. / self.tsteps / 5. * (-self.MDpreTrace + rout)
        self.MDpostTrace += 1. / self.tsteps / 5. * (-self.MDpostTrace + MDout)
        MDoutTrace = self.winner_take_all(self.MDpostTrace)

        return MDoutTrace

    def update_weights(self, rout, MDout):
        """Update weights with plasticity rules.

        Args:
            rout: input to MD
            MDout: activity of MD
        """
        MDoutTrace = self.update_trace(rout, MDout)

        self.MDpreTrace_threshold = np.mean(self.MDpreTrace)
        
        MDoutTrace_threshold = 0.5  
        
        # update and clip the weights
        # original
        wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)

        self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        self.wMD2PFC = np.clip(self.wMD2PFC + (wPFC2MDdelta.T), -10., 0.)
        self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 0.1*(wPFC2MDdelta.T), 0.,7. / self.G)
        
        # slow-decaying PFC-MD weights
        # wPFC2MDdelta = 30000 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)

        # self.wPFC2MD += 1. / self.tsteps / 5. * (-1.0 * self.wPFC2MD + 1.0 * wPFC2MDdelta)
        # self.wPFC2MD = np.clip(self.wPFC2MD, 0., 1.)
        # self.wMD2PFC += 1. / self.tsteps / 5. * (-1.0 * self.wMD2PFC + 1.0 * (wPFC2MDdelta.T))
        # self.wMD2PFC = np.clip(self.wMD2PFC, -10., 0.)
        # self.wMD2PFCMult += 1. / self.tsteps / 5. * (-1.0 * self.wMD2PFCMult + 1.0 * (wPFC2MDdelta.T))
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult, 0.,7. / self.G)
        
        # decaying PFC-MD weights
        # alpha = 0 # 0.5 when shift on, 0 when shift off
        # self.wPFC2MD = np.clip((1-alpha)* self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        # self.wMD2PFC = np.clip((1-alpha) * self.wMD2PFC + (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip((1-alpha) * self.wMD2PFCMult + (wPFC2MDdelta.T), 0.,7. / self.G)

    def winner_take_all(self, MDinp):
        '''Winner take all on the MD
        '''

        # Thresholding
        MDout = np.zeros(self.Num_MD)
        MDinp_sorted = np.sort(MDinp)

        MDthreshold = np.mean(MDinp_sorted[-int(self.num_active) * 2:])
        # MDthreshold  = np.mean(MDinp)
        index_pos = np.where(MDinp >= MDthreshold)
        index_neg = np.where(MDinp < MDthreshold)
        MDout[index_pos] = 1
        MDout[index_neg] = 0

        return MDout


class PytorchPFCMD(nn.Module):
    def __init__(self, Ntasks, input_size_per_task, Num_PFC, n_neuron_per_cue, Num_MD, num_active, num_output, MDeffect=True, noisePresent = False, noiseInput = False):
        super().__init__()
        """
        additional noise input neuron if noiseInput is true
        """
        dt = 0.001
        if noiseInput==False:
            self.sensory2pfc = SensoryInputLayer(
                Ntasks=Ntasks,
                input_size_per_task=input_size_per_task,
                n_output=Num_PFC,
                n_sub=n_neuron_per_cue)
            self.sensory2pfc.torch(use_torch=True)
            # try learnable input weights
            # self.PytorchSensory2pfc = nn.Linear(4, Num_PFC)
        # unchanged for neurogym tasks
        else:
            raise NotImplementedError
        #     self.sensory2pfc = SensoryInputLayer_NoiseNeuro(
        #         n_sub=n_neuron_per_cue,
        #         n_cues=4,
        #         n_output=Num_PFC)
        #     self.sensory2pfc.torch(use_torch=True)

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

        # initialize variables for saving important network activities
        self.pfc_outputs = torch.zeros((n_time, self.pfc.Nneur))
        self.md_preTraces = np.zeros(shape=(n_time, self.pfc.Nneur))
        self.md_preTrace_thresholds = np.zeros(shape=(n_time, 1))
        
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
            
            if self.MDeffect:
                self.md_output = self.md(pfc_output.cpu().detach().numpy())

                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.pfc.Jrec.cpu().detach().numpy(), self.pfc.activity.cpu().detach().numpy())
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.num_output))
                md2pfc = md2pfc_weights * rec_inp  
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.num_output), self.md_output)
                #pfc_output = self.pfc(torch.from_numpy(input2pfc), torch.from_numpy(md2pfc)).numpy()
                
                pfc_output = self.pfc(input2pfc,torch.from_numpy(md2pfc))
                
                # save important network activities
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
                self.md_preTraces[i, :] = self.md.MDpreTrace
                self.md_preTrace_thresholds[i, :] = self.md.MDpreTrace_threshold
                
#                pfc_output = self.pfc(input2pfc, torch.from_numpy(md2pfc)).cpu().detach().numpy()
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
                
        ## manually shut down context-irrelevant pc activity
        #import pdb;pdb.set_trace() 
#        if input[0,0]==1 or input[200,0]==1:
#            self.pfc_outputs[:,400:] *= 0
#        else:
#            self.pfc_outputs[:,:400] *= 0
#            self.pfc_outputs[:,800:] *= 0
            
        outputs = self.pfc2out(self.pfc_outputs)
        # outputs = torch.tanh(outputs)

            
        return outputs

    def _check_shape(self, input, target):
        assert len(input.shape) == self.num_output
        assert len(target.shape) == 2
        assert input.shape[0] == target.shape[0]


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
            # keep Elman activities positive
            # self.activation = lambda inp: torch.clip(torch.tanh(inp), 0, None)

        # Sensory input -> RNN
        # self.input2h = nn.Linear(input_size, hidden_size, bias=False)
        # keep sensory input layer's weights positive
        # nn.init.uniform_(self.input2h.weight, a=0.5, b=1.0)
        self.input2h = nn.Linear(input_size, hidden_size)
        k = (1./self.hidden_size)**0.5
        nn.init.uniform_(self.input2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        nn.init.uniform_(self.input2h.bias, a=-k, b=k)

        # RNN -> RNN
        # self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.reset_parameters()
        self.h2h = nn.Linear(hidden_size, hidden_size)
        k = (1./self.hidden_size)**0.5
        nn.init.uniform_(self.h2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        nn.init.uniform_(self.h2h.bias, a=-k, b=k)

    # def reset_parameters(self):
    #     '''
    #     Reset RNN weights
    #     '''
    #     #nn.init.eye_(self.h2h.weight)
    #     #self.h2h.weight.data *= 0.5

    #     #nn.init.normal_(self.h2h.weight, mean=0.0, std=1.0)

    #     k = (1./self.hidden_size)**0.5
    #     nn.init.uniform_(self.h2h.weight, a=-k, b=k) # same as pytorch built-in RNN module
        
    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(1, batch_size, self.hidden_size)

    # def recurrence(self, input, hidden, mdinput):
    #     '''
    #     Recurrence helper function
    #     '''
    #     pre_activation = self.input2h(input) + self.h2h(hidden)

    #     if mdinput is not None:
    #         pre_activation += mdinput
        
    #     h_new = self.activation(pre_activation)

    #     return h_new

    def forward(self, input, hidden=None, mdinput=None):
        '''
        Propogate input through the network
        '''
        # TODO: input.shape has to be [timestep=1, batch_size, input_size]

        # if hidden is None:
        #     hidden = self.init_hidden(input)

        # output = []
        # steps = range(input.size(0))
        # for i in steps:
        #     hidden = self.recurrence(input[i], hidden, mdinput)
        #     output.append(hidden)
        # output = torch.stack(output, dim=0)

        # hidden = self.recurrence(input[0], hidden, mdinput)

        if hidden is None:
            hidden = self.init_hidden(input)

        pre_activation = self.input2h(input) + self.h2h(hidden)

        if mdinput is not None:
            pre_activation += mdinput
        
        hidden = self.activation(pre_activation)
        
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

        # Track model parameters
        self.parm = dict()
        for name, param in self.named_parameters():
            self.parm[name] = param

    def forward(self, input, target):
        '''
        Propogate input through the network
        '''
        n_time = input.shape[0]
        batch_size = input.shape[1]

        # initialize variables for saving important network activities
        RNN_output = torch.zeros((n_time, batch_size, self.hidden_size))
        if self.MDeffect:
            self.md_preTraces = np.zeros(shape=(n_time, self.hidden_size))
            self.md_preTrace_thresholds = np.zeros(shape=(n_time, 1))
            self.md_output_t *= 0

        # initialize RNN and MD activities
        RNN_hidden_t = torch.zeros((batch_size, self.hidden_size))
        if self.MDeffect:
            self.md.init_activity()  # Reinit MD activity
        

        for t in range(n_time):
            input_t = input[t, ...].unsqueeze(dim=0)
            target_t = target[t, ...].unsqueeze(dim=0)
            
            # Reinit MD activity for each trial
            if t % self.tsteps == 0: 
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            if self.MDeffect:
                # Generate MD activities
                self.md_output = self.md(RNN_hidden_t.cpu().detach().numpy()[0, :]) # batch size should be 1
                
                # Generate MD -> PFC inputs
                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.parm['rnn.h2h.weight'].cpu().detach().numpy(), RNN_hidden_t.cpu().detach().numpy()[0, :])  # PFC recurrent inputs # batch size should be 1
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.output_size))
                md2pfc = md2pfc_weights * rec_inp                                                                # MD inputs - multiplicative term
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.output_size), self.md_output)    # MD inputs - additive term
                md2pfc = torch.from_numpy(md2pfc).view_as(RNN_hidden_t)
                
                # Generate RNN activities
                RNN_hidden_t = self.rnn(input_t, RNN_hidden_t, md2pfc)
                RNN_output[t, :, :] = RNN_hidden_t

                # save important network activities
                self.md_preTraces[t, :] = self.md.MDpreTrace
                self.md_preTrace_thresholds[t, :] = self.md.MDpreTrace_threshold
                
                # Collect MD activities
                if t==0:
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)

            else:
                RNN_hidden_t = self.rnn(input_t, RNN_hidden_t)
                RNN_output[t, :, :] = RNN_hidden_t

        model_out = self.fc(RNN_output)
        model_out = torch.tanh(model_out)

        return model_out

# Elman or LSTM
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        # self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        # out, hidden = self.rnn(x)
        x = self.linear(out)
        return x, out

# CTRNN model
# origin https://github.com/neurogym/ngym_usage/tree/master/yang19
class CTRNN(nn.Module):
    """Continuous-time RNN.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        # initialized as an identity matrix*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

        # the same as pytorch built-in RNN module
        # k = (1./self.hidden_size)**0.5
        # nn.init.uniform_(self.h2h.weight, a=-k, b=k)
        # nn.init.uniform_(self.h2h.bias, a=-k, b=k)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        pre_activation = self.input2h(input) + self.h2h(hidden)
        h_new = torch.relu(hidden * self.oneminusalpha +
                           pre_activation * self.alpha)
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden

class RNNNet(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        out = self.fc(rnn_activity)
        return out, rnn_activity


# MD for neurogym tasks
class MD_GYM():
    def __init__(self, Nneur, Num_MD, num_active=1, positiveRates=True, dt=0.001):
        self.Nneur = Nneur
        self.Num_MD = Num_MD
        self.positiveRates = positiveRates
        self.num_active = num_active # num_active: num MD active per context
        self.learn = True # update MD weights or not
        self.sendinputs = True # send inputs to RNN or not

        self.tau = 0.02
        self.tau_times = 4
        self.dt = dt
        self.tau_trace = 500 # unit, time steps
        self.Hebb_learning_rate = 1e-4
        Gbase = 0.75  # determines also the cross-task recurrence

        # initialize weights
        self.wPFC2MD = np.random.normal(0,
                                        1 / np.sqrt(self.Num_MD * self.Nneur),
                                        size=(self.Num_MD, self.Nneur))
        self.wMD2PFC = np.random.normal(0,
                                        1 / np.sqrt(self.Num_MD * self.Nneur),
                                        size=(self.Nneur, self.Num_MD))
        self.wMD2PFCMult = np.random.normal(0,
                                            1 / np.sqrt(self.Num_MD * self.Nneur),
                                            size=(self.Nneur, self.Num_MD))
        # initialize activities
        self.prev_PFCout = np.zeros(shape=(self.Nneur)) # PFC activities in the previous step
        self.MDpreTrace = np.zeros(shape=(self.Nneur))
        self.MDpostTrace = np.zeros(shape=(self.Num_MD))
        self.MDpreTrace_threshold = 0

        # Choose G based on the type of activation function
        #  unclipped activation requires lower G than clipped activation,
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
        self.prev_PFCout = np.zeros(shape=(self.Nneur))
        
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
        # compute MD outputs
        #  MD decays 10x slower than PFC neurons,
        #  so as to somewhat integrate PFC input over time
        if self.positiveRates:
            self.MDinp += self.dt / self.tauMD * (-self.MDinp + np.dot(self.wPFC2MD, input))
        else:
            # shift PFC rates, so that mean is non-zero to turn MD on
            self.MDinp += self.dt / self.tauMD * (-self.MDinp + np.dot(self.wPFC2MD, (input + 0.5)))      
        MDout = self.winner_take_all(self.MDinp)

        # update
        if self.learn:
            # update PFC-MD weights
            self.update_weights(input, MDout)
            # update PFC activities in the previous step
            self.prev_PFCout = input

        return MDout

    def update_trace(self, rout, MDout):
        # update pretrace based on the difference between steps
        self.MDpreTrace += 1. / self.tau_trace * (-self.MDpreTrace + 2*abs(rout - self.prev_PFCout))
        self.MDpostTrace += 1. / self.tau_trace * (-self.MDpostTrace + MDout)
        MDoutTrace = self.winner_take_all(self.MDpostTrace)

        return MDoutTrace

    def update_weights(self, rout, MDout):
        """Update weights with plasticity rules.

        Args:
            rout: input to MD
            MDout: activity of MD
        """
        
        MDoutTrace = self.update_trace(rout, MDout)

        self.MDpreTrace_threshold = np.mean(self.MDpreTrace)
        MDoutTrace_threshold = 0.5
        
        # update and clip the weights
        #  original
        wPFC2MDdelta = 15 * 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold, self.MDpreTrace - self.MDpreTrace_threshold)
        self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        self.wMD2PFC = np.clip(self.wMD2PFC + (wPFC2MDdelta.T), -10., 0.)
        self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 0.1*(wPFC2MDdelta.T), 0.,7. / self.G)
        
        #  slow-decaying PFC-MD weights
        # wPFC2MDdelta = 30000 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)
        # self.wPFC2MD += 1. / self.tsteps / 5. * (-1.0 * self.wPFC2MD + 1.0 * wPFC2MDdelta)
        # self.wPFC2MD = np.clip(self.wPFC2MD, 0., 1.)
        # self.wMD2PFC += 1. / self.tsteps / 5. * (-1.0 * self.wMD2PFC + 1.0 * (wPFC2MDdelta.T))
        # self.wMD2PFC = np.clip(self.wMD2PFC, -10., 0.)
        # self.wMD2PFCMult += 1. / self.tsteps / 5. * (-1.0 * self.wMD2PFCMult + 1.0 * (wPFC2MDdelta.T))
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult, 0.,7. / self.G)
        
        #  decaying PFC-MD weights
        # alpha = 0 # 0.5 when shift on, 0 when shift off
        # self.wPFC2MD = np.clip((1-alpha)* self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        # self.wMD2PFC = np.clip((1-alpha) * self.wMD2PFC + (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip((1-alpha) * self.wMD2PFCMult + (wPFC2MDdelta.T), 0.,7. / self.G)

    def winner_take_all(self, MDinp):
        '''Winner take all on the MD
        '''

        # Thresholding
        MDout = np.zeros(self.Num_MD)
        MDinp_sorted = np.sort(MDinp)

        MDthreshold = np.mean(MDinp_sorted[-int(self.num_active) * 2:])
        # MDthreshold  = np.mean(MDinp)
        index_pos = np.where(MDinp >= MDthreshold)
        index_neg = np.where(MDinp < MDthreshold)
        MDout[index_pos] = 1
        MDout[index_neg] = 0

        return MDout

# CTRNN with MD layer
class CTRNN_MD(nn.Module):
    """Continuous-time RNN that can take MD inputs.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        sub_size: Number of subpopulation neurons
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, sub_size, output_size, MDeffect, md_size, md_active_size, md_dt, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_size = sub_size
        self.output_size = output_size
        self.md_size = md_size

        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        # sensory input layer
        self.input2h = nn.Linear(input_size, sub_size)

        # hidden layer
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.reset_parameters()

        # MD layer
        self.MDeffect = MDeffect
        if self.MDeffect:
            self.md = MD_GYM(Nneur=hidden_size, Num_MD=md_size, num_active=md_active_size, dt=md_dt, positiveRates=True)
            self.md.md_output = np.zeros(md_size)
            index = np.random.permutation(md_size)
            self.md.md_output[index[:md_active_size]] = 1 # randomly set part of md_output to 1
            self.md.md_output_t = np.array([])

    def reset_parameters(self):
        # initialized as an identity matrix*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

        # the same as pytorch built-in RNN module
        # used in reservoir
        # k = (1./self.hidden_size)**0.5
        # nn.init.uniform_(self.h2h.weight, a=-k, b=k)
        # nn.init.uniform_(self.h2h.bias, a=-k, b=k)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)

    def recurrence(self, input, sub_id, hidden):
        """Recurrence helper."""
        ext_input = self.input2h(input)
        rec_input = self.h2h(hidden)

        # expand inputs
        ext_input_expanded = torch.zeros_like(rec_input)
        ext_input_expanded[:, sub_id*self.sub_size:(sub_id+1)*self.sub_size] = ext_input

        pre_activation = ext_input_expanded + rec_input

        # md inputs
        if self.MDeffect:
            assert hidden.shape[0] == 1, 'batch size should be 1'
            assert rec_input.shape[0] == 1, 'batch size should be 1'

            self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])

            self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md.md_output)
            rec_inp = rec_input.cpu().detach().numpy()[0, :]
            md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.output_size))
            md2pfc = md2pfc_weights * rec_inp
            md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.output_size), self.md.md_output)
            md2pfc = torch.from_numpy(md2pfc).view_as(hidden).to(input.device)

            if self.md.sendinputs:
                pre_activation += md2pfc
        
        h_new = torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)
        
        # shutdown analysis
        # shutdown_mask = torch.zeros_like(h_new)
        # shutdown_mask[:, sub_id*self.sub_size:(sub_id+1)*self.sub_size] = 1
        # h_new = h_new.mul(shutdown_mask)

        return h_new

    def forward(self, input, sub_id, hidden=None):
        """Propogate input through the network."""
        
        num_tsteps = input.size(0)

        # init network activities
        if hidden is None:
            hidden = self.init_hidden(input)
        if self.MDeffect:
            self.md.init_activity()

        # initialize variables for saving network activities
        output = []
        if self.MDeffect:
            self.md.md_preTraces = np.zeros(shape=(num_tsteps, self.hidden_size))
            self.md.md_preTrace_thresholds = np.zeros(shape=(num_tsteps, 1))
            self.md.md_output_t *= 0

        for i in range(num_tsteps):
            hidden = self.recurrence(input[i], sub_id, hidden)
            
            # save PFC activities
            output.append(hidden)
            # save MD activities
            if self.MDeffect:
                self.md.md_preTraces[i, :] = self.md.MDpreTrace
                self.md.md_preTrace_thresholds[i, :] = self.md.MDpreTrace_threshold
                if i==0:
                    self.md.md_output_t = self.md.md_output.reshape((1, self.md.md_output.shape[0]))
                else:
                    self.md.md_output_t = np.concatenate((self.md.md_output_t, self.md.md_output.reshape((1, self.md.md_output.shape[0]))),axis=0)

        output = torch.stack(output, dim=0)
        return output, hidden

class RNN_MD(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        sub_size: int, subpopulation size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, input_size, hidden_size, sub_size, output_size, MDeffect, md_size, md_active_size, md_dt, **kwargs):
        super().__init__()

        self.rnn = CTRNN_MD(input_size, hidden_size, sub_size, output_size, MDeffect, md_size, md_active_size, md_dt, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, sub_id):
        rnn_activity, _ = self.rnn(x, sub_id)
        out = self.fc(rnn_activity)
        return out, rnn_activity