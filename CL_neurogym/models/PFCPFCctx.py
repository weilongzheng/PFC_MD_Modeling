import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class CTRNN_PFCctx(nn.Module):
    """Continuous-time RNN that can take MD inputs.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        sub_size: Number of subpopulation neurons
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, hidden_ctx_size, sub_size, sub_active_size, output_size, config, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_ctx_size = hidden_ctx_size
        self.sub_size = sub_size
        self.sub_active_size = sub_active_size
        self.output_size = output_size
        self.config = config
        

        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        # sensory input -> PFC
        self.input2h = nn.Linear(input_size, hidden_size)
        # PFC -> PFC
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # sensory input -> PFC context
        self.input2PFCctx = nn.Linear(input_size, hidden_ctx_size, bias=False)
        # PFC context -> PFC
        '''
        Replace the MD with a pretrained linear PFCctx2PFC layer.

        '''
        self.PFCctx2PFC = nn.Linear(hidden_ctx_size, hidden_size, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        ### input to PFCctx weights initialization
        # all uniform noise
        k = (1./self.hidden_size)**0.5
        self.input2PFCctx.weight.data = k*torch.rand(self.input2PFCctx.weight.data.size()) # noise ~ U(leftlim=0, rightlim=k)

        ### PFCctx to PFC weights initialization (the same as MD to PFC initialization)
        weight_data = torch.zeros(self.hidden_size, self.hidden_ctx_size)
        for i in range(weight_data.shape[0]):
            j = np.floor(np.random.rand()*self.hidden_ctx_size).astype(int)
            weight_data[i, j] = -5 # original -5
        self.PFCctx2PFC.weight.data = weight_data

        ### PFC recurrent weights initialization
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

        # default initialization
        # pass

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

        pre_activation = ext_input + rec_input

        # external inputs & activities of PFC neurons containing context info
        ext_input_ctx = self.input2PFCctx(input)
        # PFC-context neurons get disjoint inputs
        # 1. The context information is not deterministic
        # 2. The PFC-context layer is noisy
        ext_input_mask = torch.zeros_like(ext_input_ctx)
        mask_idx = torch.where(torch.rand(self.sub_size) < self.config.sub_active_prob)[0].tolist()
        for batch_idx in range(ext_input_mask.shape[0]):
            ext_input_mask[batch_idx, sub_id*self.sub_size:(sub_id+1)*self.sub_size][mask_idx] = 1
        PFC_ctx_input = torch.relu(ext_input_ctx.mul(ext_input_mask) + (self.config.hidden_ctx_noise)*torch.randn(ext_input_ctx.size()))
        # save PFC-ctx activity
        self.PFC_ctx_act = PFC_ctx_input

        # PFCctx outputs
        PFC_ctx_output = self.PFCctx2PFC(PFC_ctx_input)
        self.PFC_ctx_out = PFC_ctx_output

        pre_activation += PFC_ctx_output
        
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

        # initialize variables for saving network activities
        output = []
        self.PFC_ctx_acts = np.zeros(shape=(num_tsteps, self.hidden_ctx_size))

        for i in range(num_tsteps):
            hidden = self.recurrence(input[i], sub_id, hidden)
            
            # save PFC activities
            output.append(hidden)
            self.PFC_ctx_acts[i, :] = self.PFC_ctx_act.detach().numpy()

        output = torch.stack(output, dim=0)
        return output, hidden

class RNN_PFCctx(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        sub_size: int, subpopulation size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, input_size, hidden_size, hidden_ctx_size, sub_size, sub_active_size, output_size, config, **kwargs):
        super().__init__()

        self.rnn = CTRNN_PFCctx(input_size, hidden_size, hidden_ctx_size, sub_size, sub_active_size, output_size, config, **kwargs)
        self.drop_layer = nn.Dropout(p=0.0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, task_id):
        rnn_activity, _ = self.rnn(x, sub_id=task_id)
        rnn_activity = self.drop_layer(rnn_activity)
        out = self.fc(rnn_activity)
        return out, rnn_activity