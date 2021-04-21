# this Elman_MD class is based on pytorch built-in RNN
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