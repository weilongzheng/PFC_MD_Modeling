# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys, shelve

try:
    import torch
    from torch import nn
except ImportError:
    print('Torch not available')


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
        self.tau_pretrace = 1000 # unit, time steps
        self.tau_posttrace = 1000 # unit, time steps
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
        self.MDpreTrace_binary = np.zeros(shape=(self.Nneur))
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

        return MDout

    def update_trace(self, rout, MDout):
        self.MDpreTrace += 1. / self.tau_pretrace * (-self.MDpreTrace + rout)
        self.MDpostTrace += 1. / self.tau_posttrace * (-self.MDpostTrace + MDout)
        MDoutTrace = self.winner_take_all(self.MDpostTrace)

        return MDoutTrace

    def update_weights(self, rout, MDout):
        """Update weights with plasticity rules.

        Args:
            rout: input to MD
            MDout: activity of MD
        """

        # MD outputs
        MDoutTrace = self.update_trace(rout, MDout)

        # use OR opertion to get binary pretraces
        pretrace_part = int(1.0*len(self.MDpreTrace))
        self.MDpreTrace_threshold = np.mean(np.sort(self.MDpreTrace)[-pretrace_part:])
        # self.MDpreTrace_threshold = np.median(np.sort(self.MDpreTrace)[-pretrace_part:])
        # self.MDpreTrace_binary = (self.MDpreTrace>self.MDpreTrace_threshold).astype(float)

        # compute thresholds
        # self.MDpreTrace_binary_threshold = np.mean(self.MDpreTrace_binary)
        self.MDpreTrace_binary_threshold = 0.5
        MDoutTrace_threshold = 0.5
        
        # update and clip the PFCMD weights
        wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold, self.MDpreTrace_binary - self.MDpreTrace_binary_threshold)
        self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)
        self.wMD2PFC = np.clip(self.wMD2PFC + wPFC2MDdelta.T, -1, 0.)
        self.wMD2PFCMult = np.clip(self.wMD2PFCMult + wPFC2MDdelta.T, 0., 1.)

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

    def __init__(self, input_size, hidden_size, sub_size, output_size, num_task, MDeffect, md_size, md_active_size, md_dt, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sub_size = sub_size
        self.output_size = output_size
        self.num_task = num_task
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
        # identity*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

        # identity*other value
        # nn.init.eye_(self.h2h.weight)
        # self.h2h.weight.data *= 0.2

        # block identity + positive uniform noise
        # weights = []
        # for i in range(self.num_task):
        #     k = 1e-1*(1./self.hidden_size)**0.5
        #     weights.append(torch.eye(self.sub_size)*0.5 + k*torch.rand(self.sub_size, self.sub_size)) # noise ~ U(leftlim=0, rightlim=k)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # block identity + uniform noise
        # weights = []
        # for i in range(self.num_task):
        #     k = (1./self.hidden_size)**0.5
        #     weights.append(torch.eye(self.sub_size) + 2*k*torch.rand(self.sub_size, self.sub_size) - k) # noise ~ U(leftlim=-k, rightlim=k)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # block identity + normal noise
        # weights = []
        # for i in range(self.num_task):
        #     k = (1./self.hidden_size)**0.5
        #     weights.append(torch.eye(self.sub_size) + k*torch.randn(self.sub_size, self.sub_size)) # noise ~ N(mean=0, std=1/hidden_size)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # block positive uniform noise
        # weights = []
        # for i in range(self.num_task):
        #     k = 1e-1*(1./self.hidden_size)**0.5
        #     weights.append(k*torch.rand(self.sub_size, self.sub_size)) # noise ~ U(leftlim=0, rightlim=k)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # random orthogonal noise
        # nn.init.orthogonal_(self.h2h.weight, gain=0.5)

        # all uniform noise
        # k = (1./self.hidden_size)**0.5
        # self.h2h.weight.data += 2*k*torch.rand(self.h2h.weight.data.size()) - k # noise ~ U(leftlim=-k, rightlim=k)

        # all normal noise
        # k = (1./self.hidden_size)**0.5
        # self.h2h.weight.data += k*torch.randn(self.h2h.weight.data.size()) # noise ~ N(mean=0, std=1/hidden_size)

        # the same as pytorch built-in RNN module, used in reservoir
        # k = (1./self.hidden_size)**0.5
        # nn.init.uniform_(self.h2h.weight, a=-k, b=k)
        # nn.init.uniform_(self.h2h.bias, a=-k, b=k)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        # as zeros
        hidden = torch.zeros(batch_size, self.hidden_size)
        # as uniform noise
        # hidden = 1/self.hidden_size*torch.rand(batch_size, self.hidden_size)
        return hidden.to(input.device)

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

            # original MD inputs
            # self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])
            # self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md.md_output)
            # rec_inp = rec_input.cpu().detach().numpy()[0, :]
            # md2pfc_weights = (self.md.MD2PFCMult/self.md.Num_MD)
            # md2pfc = md2pfc_weights * rec_inp
            # md2pfc += np.dot((self.md.wMD2PFC/self.md.Num_MD), self.md.md_output)
            # md2pfc = torch.from_numpy(md2pfc).view_as(hidden).to(input.device)

            # only MD additive inputs
            # self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])
            # md2pfc = np.dot((self.md.wMD2PFC/self.md.Num_MD), self.md.md_output)
            # md2pfc = torch.from_numpy(md2pfc).view_as(hidden).to(input.device)

            # ideal MD inputs analysis
            # self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])
            # rec_inp = rec_input.cpu().detach().numpy()[0, :]
            # #  ideal multiplicative inputs
            # md2pfc_weights = np.zeros(shape=(self.hidden_size))
            # md2pfc_weights[sub_id*self.sub_size:(sub_id+1)*self.sub_size] = 0.5
            # md2pfcMult = md2pfc_weights * rec_inp
            # #  ideal additive inputs
            # md2pfcAdd = np.ones(shape=(self.hidden_size))*(-0.5)
            # md2pfcAdd[sub_id*self.sub_size:(sub_id+1)*self.sub_size] = 0
            # #  ideal inputs
            # md2pfc = md2pfcAdd
            # md2pfc = torch.from_numpy(md2pfc).view_as(hidden).to(input.device)

            # hard-coded MD pretraces
            #  remember to turn off the update of MDpreTrace_binary in MD class
            # perfect pretraces
            self.md.MDpreTrace_binary = np.zeros(shape=(self.hidden_size))
            self.md.MDpreTrace_binary[sub_id*self.sub_size:(sub_id+1)*self.sub_size] = 1
            # imperfect pretraces
            # wrong_ratio = 0.4
            # flag = np.random.rand(self.hidden_size)
            # self.md.MDpreTrace_binary = (flag < wrong_ratio).astype(float)
            # flag = np.random.rand(self.sub_size)
            # self.md.MDpreTrace_binary[sub_id*self.sub_size:(sub_id+1)*self.sub_size] = (flag > wrong_ratio).astype(float)

            self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])
            self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md.md_output)
            rec_inp = rec_input.cpu().detach().numpy()[0, :]
            md2pfc_weights = (self.md.MD2PFCMult/self.md.Num_MD)
            md2pfcMult = md2pfc_weights * rec_inp
            md2pfcAdd  = np.dot((self.md.wMD2PFC/self.md.Num_MD), self.md.md_output)
            # md2pfc = md2pfcAdd + md2pfcMult
            md2pfc = md2pfcAdd
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
            self.md.md_preTraces_binary = np.zeros(shape=(num_tsteps, self.hidden_size))
            self.md.md_preTrace_thresholds = np.zeros(shape=(num_tsteps, 1))
            self.md.md_preTrace_binary_thresholds = np.zeros(shape=(num_tsteps, 1))
            self.md.md_output_t *= 0

        for i in range(num_tsteps):
            hidden = self.recurrence(input[i], sub_id, hidden)
            
            # save PFC activities
            output.append(hidden)
            # save MD activities
            if self.MDeffect:
                self.md.md_preTraces[i, :] = self.md.MDpreTrace
                self.md.md_preTraces_binary[i, :] = self.md.MDpreTrace_binary
                self.md.md_preTrace_thresholds[i, :] = self.md.MDpreTrace_threshold
                self.md.md_preTrace_binary_thresholds[i, :] = self.md.MDpreTrace_binary_threshold
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

    def __init__(self, input_size, hidden_size, sub_size, output_size, num_task, MDeffect, md_size, md_active_size, md_dt, **kwargs):
        super().__init__()

        self.rnn = CTRNN_MD(input_size, hidden_size, sub_size, output_size, num_task, MDeffect, md_size, md_active_size, md_dt, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, sub_id):
        rnn_activity, _ = self.rnn(x, sub_id)
        out = self.fc(rnn_activity)
        return out, rnn_activity