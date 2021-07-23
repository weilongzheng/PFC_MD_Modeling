import numpy as np
import matplotlib.pyplot as plt
import sys, shelve

try:
    import torch
    from torch import nn
except ImportError:
    print('Torch not available')


class SensoryInputLayer():
    def __init__(self, n_sub, n_cues, n_output):
        self.Ncues = n_cues
        self.Nsub = n_sub
        self.Nneur = n_output
        self.positiveRates = True
        self.weightNoise = False
        self.weightOverlap = False

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
        # add random noise to input weights
        if self.weightNoise==True:
            noiseSD = 1e-1
            self.wIn += np.random.normal(size=(np.shape(self.wIn))) * noiseSD
        
        # Input weights have overlops (mix neurons)
        if self.weightOverlap == True:
            ''' overlap across rules'''
            for cuei in np.arange(self.Ncues):
                self.wIn[self.Nsub * cuei:self.Nsub * (cuei + 1)+int(self.Nsub/2), cuei] = \
                    np.random.uniform(lowcue, highcue, size=self.Nsub+int(self.Nsub/2)) \
                    * self.cueFactor
            
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


class PytorchPFC(nn.Module):
    def __init__(self, n_neuron, n_neuron_per_cue, pfcNoise, positiveRates=True, MDeffect=True, noisePresent = False):
        super().__init__()
        self.Nneur = n_neuron
        self.Nsub = n_neuron_per_cue
        self.useMult = True
        self.noisePresent = noisePresent
        self.noiseSD = pfcNoise #1e-2  # 1e-3
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
        
        # Update and clip the weights
        # original
        # wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)
        # self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        # self.wMD2PFC = np.clip(self.wMD2PFC + (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 0.1*(wPFC2MDdelta.T), 0.,7. / self.G)
        
        # Decaying PFC-MD weights version 1
        '''
        Increase the Hebbian learning rate because we want to PFC-MD weights are large enough (we use ODE here).
        '''
        wPFC2MDdelta = 30000 * 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold, self.MDpreTrace - self.MDpreTrace_threshold)
        self.wPFC2MD += 1. / self.tsteps / 5. * (-1.0 * self.wPFC2MD + 1.0 * wPFC2MDdelta)
        self.wPFC2MD = np.clip(self.wPFC2MD, 0., 1.)
        self.wMD2PFC += 1. / self.tsteps / 5. * (-1.0 * self.wMD2PFC + 1.0 * (wPFC2MDdelta.T))
        self.wMD2PFC = np.clip(self.wMD2PFC, -10., 0.)
        self.wMD2PFCMult += 1. / self.tsteps / 5. * (-1.0 * self.wMD2PFCMult + 1.0 * (wPFC2MDdelta.T))
        self.wMD2PFCMult = np.clip(self.wMD2PFCMult, 0.,7. / self.G)
        
        # Decaying PFC-MD weights version 2
        # alpha = 0.01 # 0.5 when shift on, 0 when shift off
        # wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)
        # self.wPFC2MD = np.clip((1-alpha)* self.wPFC2MD + alpha*wPFC2MDdelta, 0., 1.)
        # self.wMD2PFC = np.clip((1-alpha) * self.wMD2PFC + alpha*(wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip((1-alpha) * self.wMD2PFCMult + alpha*(wPFC2MDdelta.T), 0.,7. / self.G)

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
    def __init__(self, Num_PFC, n_neuron_per_cue, n_cues, Num_MD, num_active, num_output, pfcNoise, MDeffect=True, noisePresent = False, noiseInput = False):
        super().__init__()
        """
        additional noise input neuron if noiseInput is true
        """
        dt = 0.001
        if noiseInput==False:
            self.sensory2pfc = SensoryInputLayer(
                n_sub=n_neuron_per_cue,
                n_cues=n_cues,
                n_output=Num_PFC)
            self.sensory2pfc.torch(use_torch=True)
            # try learnable input weights
            # self.PytorchSensory2pfc = nn.Linear(4, Num_PFC)
        else:
            self.sensory2pfc = SensoryInputLayer_NoiseNeuro(
                n_sub=n_neuron_per_cue,
                n_cues=n_cues,
                n_output=Num_PFC)
            self.sensory2pfc.torch(use_torch=True)

        self.pfc = PytorchPFC(Num_PFC, n_neuron_per_cue, pfcNoise, MDeffect=MDeffect, noisePresent = noisePresent)

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
            self.md_output_t *= 0

        #output = torch.zeros((n_time, target.shape[-1]))
        #self.pfc_output_t *= 0

        # initialize variables for saving important network activities
        self.pfc_outputs = torch.zeros((n_time, self.pfc.Nneur))
        if self.MDeffect:
            self.md_preTraces = np.zeros(shape=(n_time, self.pfc.Nneur))
            self.md_preTrace_thresholds = np.zeros(shape=(n_time, 1))
            self.wPFC2MDs_all = np.zeros(shape=(n_time,self.md.Num_MD,self.pfc.Nneur))
            self.wMD2PFCs_all = np.zeros(shape=(n_time,self.pfc.Nneur,self.md.Num_MD))

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
                #pfc_output = self.pfc(torch.from_numpy(input2pfc), torch.from_numpy(md2pfc)).numpy()
                
                pfc_output = self.pfc(input2pfc,torch.from_numpy(md2pfc))
                
                # save important network activities
                pfc_output_t = pfc_output.view(1,pfc_output.shape[0])
                self.pfc_outputs[i, :] = pfc_output_t
                self.md_preTraces[i, :] = self.md.MDpreTrace
                self.md_preTrace_thresholds[i, :] = self.md.MDpreTrace_threshold
                self.wPFC2MDs_all[i,:,:] = torch.from_numpy(self.md.wPFC2MD)
                self.wMD2PFCs_all[i,:,:] = torch.from_numpy(self.md.wMD2PFC)
                
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
                
        ## manually shut down context-irrelevant pc activity
        #import pdb;pdb.set_trace() 
#        if input[0,0]==1 or input[200,0]==1:
#            self.pfc_outputs[:,400:] *= 0
#        else:
#            self.pfc_outputs[:,:400] *= 0
#            self.pfc_outputs[:,800:] *= 0
            
        outputs = self.pfc2out(self.pfc_outputs)
        outputs = torch.tanh(outputs)
            
        return outputs

    def _check_shape(self, input, target):
        assert len(input.shape) == self.num_output
        assert len(target.shape) == 2
        assert input.shape[0] == target.shape[0]