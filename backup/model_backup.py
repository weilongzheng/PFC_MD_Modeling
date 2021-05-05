#---------------- MD layer updating once a cycle with mean PFC and MD activities across one cycle ----------------#
class MD():
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

        # do not update within a time step
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
        wPFC2MDdelta = 100.0 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold,self.MDpreTrace - self.MDpreTrace_threshold)
        
        # Update and clip the weights
        # original
        # self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        # self.wMD2PFC = np.clip(self.wMD2PFC + 0.1 * (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 0.1 * (wPFC2MDdelta.T), 0.,7. / self.G)
        
        # keep wPFC2MD and wMD2PFC symmetrical
        # self.wPFC2MD = np.clip(self.wPFC2MD + 1.0 *  wPFC2MDdelta,      0., 10.)
        # self.wMD2PFC = np.clip(self.wMD2PFC + 1.0 * (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 1.0 * (wPFC2MDdelta.T), 0., 7. / self.G)

        # Increase the inhibition and decrease the excitation
        # self.wPFC2MD = np.clip(self.wPFC2MD + 0.1 * wPFC2MDdelta, 0., 1.)
        # self.wMD2PFC = np.clip(self.wMD2PFC + 1.0 * (wPFC2MDdelta.T), -10., 0.)
        # self.wMD2PFCMult = np.clip(self.wMD2PFCMult + 1.0 * (wPFC2MDdelta.T), 0.,7. / self.G)
        
        # only keep wPFC2MDdelta
        self.wPFC2MD = np.clip(1.0 * wPFC2MDdelta, 0., 1.)
        self.wMD2PFC = np.clip(1.0 * (wPFC2MDdelta.T), -10., 0.)
        self.wMD2PFCMult = np.clip(1.0 * (wPFC2MDdelta.T), 0.,7. / self.G)

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
            self.md = MD(Nneur=Num_PFC, Num_MD=Num_MD, num_active=num_active, dt=dt)
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

#---------------- PFC layer with moving averages in outputs (pytorch version) ----------------#
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
        compute moving average of PFC outputs
        rout_movingaverage = np.convolve(rout.detach().numpy(), np.ones(20)/20, mode='same')
        rout_movingaverage = torch.from_numpy(rout_movingaverage).type(torch.float)
        
        # original
        # self.activity = rout
        # return rout
        
        self.activity = rout_movingaverage
        return rout_movingaverage

#---------------- Elman_MD class based on pytorch built-in RNN ----------------#
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

        dt = 0.001 # Hard-coded for now

        # Elman RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=nonlinearity)

        # MD layer
        self.MDeffect = MDeffect
        if self.MDeffect:
            self.md = MD(Nneur=self.hidden_size, Num_MD=Num_MD, num_active=num_active, dt=dt)
            #  initialize md_output
            self.md_output = np.zeros(Num_MD)
            index = np.random.permutation(Num_MD)
            self.md_output[index[:num_active]] = 1 # randomly set part of md_output to 1
            self.md_output_t = np.array([])

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Track parameters
        self.parm=dict()
        for name, parameters in self.named_parameters():
            print(name,':',parameters.size())
            self.parm[name] = parameters

    def forward(self, input, target):
        n_time = input.shape[0]
        batch_size = input.shape[1]

        RNN_output = torch.zeros((n_time, batch_size, self.hidden_size))
        RNN_hidden_t = torch.zeros((self.num_layers, batch_size, self.hidden_size))

        for t in range(n_time):
            input_t = input[t, ...].unsqueeze(dim=0)
            target_t = target[t, ...].unsqueeze(dim=0)
            
            # Reinit MD activity for each trial
            if t % self.tsteps == 0: 
                if self.MDeffect:
                    self.md.init_activity()  # Reinit MD activity

            # TODO: integrate MD into Elman_MD
            if self.MDeffect:
                self.md_output = self.md(RNN_hidden_t.detach().numpy()[0, 0, :]) # batch size should be 1

                self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md_output)
                rec_inp = np.dot(self.parm['rnn.weight_hh_l0'].detach().numpy(), RNN_hidden_t.detach().numpy()[0, 0, :])  # PFC recurrent inputs # batch size should be 1
                md2pfc_weights = (self.md.MD2PFCMult / np.round(self.md.Num_MD / self.output_size))
                md2pfc = md2pfc_weights * rec_inp                                                                # MD inputs - multiplicative term
                md2pfc += np.dot(self.md.wMD2PFC / np.round(self.md.Num_MD /self.output_size), self.md_output)    # MD inputs - additive term

                ####print(self.parm['rnn.bias_hh_l0'])
                ####print(md2pfc)
                self.parm['rnn.bias_hh_l0'] += torch.from_numpy(md2pfc)
                ####print(self.parm['rnn.bias_hh_l0'])

                RNN_output[t, :, :], RNN_hidden_t = self.rnn(input_t, RNN_hidden_t)

                if i==0:
                    self.md_output_t = self.md_output.reshape((1,self.md_output.shape[0]))
                else:
                    self.md_output_t = np.concatenate((self.md_output_t, self.md_output.reshape((1,self.md_output.shape[0]))),axis=0)

            else:
                RNN_output[t, :, :], RNN_hidden_t = self.rnn(input_t, RNN_hidden_t)

        model_out = self.fc(RNN_output)
        return model_out