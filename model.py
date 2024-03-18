import torch
import torch.nn as nn
import snntorch as snn


class EchoSpike(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, c_y=[1e-4, -1e-4], beta=0.95, n_time_steps=100, recurrency_type='none', device='cuda', inp_thr=0.05):
        """
        Initializes the EchoSpike SNN with the given parameters.

        Args:
            num_inputs (int): The number of input units.
            num_hidden (list): A list of integers representing the number of hidden units in each layer.
            beta (float, optional): The beta value for initializing the leaky integrate and fire model. Defaults to 0.75.
            n_time_steps (int, optional): The number of time bins per segment. Defaults to 100.
            recurrency_type (string, optional): Select the type of recurrency: (none, stacked, full)
            device (string, optional): torch device
        """
        super().__init__()
        self.num_hidden = num_hidden
        self.recurrency_type = recurrency_type
        self.device = device

        # Initialized the EchoSpike layers with shapes from num_hidden and type of recurrency
        if self.recurrency_type == 'none':
            layer_inputs = [num_inputs]
            for idx_hidden in range(1, len(num_hidden)):
                layer_inputs.append(num_hidden[idx_hidden-1])
        elif self.recurrency_type == 'stacked':
            layer_inputs = [num_inputs+num_hidden[0]]
            for idx_hidden in range(1, len(num_hidden)):
                layer_inputs.append(num_hidden[idx_hidden-1] + num_hidden[idx_hidden])
        elif self.recurrency_type == 'full':
            layer_inputs = len(num_hidden)*[num_inputs+sum(num_hidden)] 
        elif self.recurrency_type == 'dt':
            # deep transition RNN: feed back last layer to first layer
            layer_inputs = [num_inputs+num_hidden[-1]]
            for idx_hidden in range(1, len(num_hidden)):
                layer_inputs.append(num_hidden[idx_hidden-1])
        elif self.recurrency_type == 'dense':
            # concatenate input and hidden state
            layer_inputs = [num_inputs]
            for idx_hidden in range(1, len(num_hidden)):
                layer_inputs.append(num_inputs + sum(num_hidden[:idx_hidden]))
        else:
            raise NotImplementedError

        self.layers = nn.ModuleList([])
        for idx_hidden in range(len(num_hidden)):
            # if beta is list, then layerwise different beta values are used
            b = beta[idx_hidden] if isinstance(beta, list) else beta
            
            self.layers.append(EchoSpike_layer(layer_inputs[idx_hidden], num_hidden[idx_hidden], b, n_time_steps=n_time_steps, c_y=c_y, inp_thr=inp_thr))

        self.hidden_state = None    

    def reset(self):
        accuracies = torch.zeros((len(self.layers), 2))
        for i, layer in enumerate(self.layers):
            accuracies[i] = layer.reset()
        self.hidden_state = None
        return accuracies

    def forward(self, inp, bf: int, freeze: list=[], inp_activity=None):
        with torch.no_grad():
            mems = len(self.layers)*[None]
            losses = torch.zeros((len(self.layers), 2), device=inp.device)
            # define input for first layer
            if self.recurrency_type not in ['none', 'dense']:
                if self.hidden_state is None:
                    # init hidden state
                    if self.recurrency_type == 'dt':
                        self.hidden_state = torch.zeros(inp.shape[0], self.num_hidden[-1], device=inp.device)
                    else:
                        self.hidden_state = [torch.zeros(inp.shape[0], self.num_hidden[i], device=inp.device) for i in range(len(self.num_hidden))]
                if self.recurrency_type == 'stacked':
                    layer_in = torch.cat((inp, self.hidden_state[0]), dim=1)
                elif self.recurrency_type == 'full':
                    layer_in = torch.cat((inp, *self.hidden_state), dim=1)
                elif self.recurrency_type == 'dt':
                    layer_in = torch.cat((inp, self.hidden_state), dim=1)

            else:
                layer_in = inp
            out_spk = []

            for idx, layer in enumerate(self.layers):
                factor = bf if not idx in freeze else 0
                spk, mem, loss = layer(layer_in, inp_activity=inp_activity)
                if idx < len(self.layers) - 1:
                    if self.recurrency_type == 'full':
                        layer_in = torch.cat((inp, spk, *self.hidden_state[idx+1:], *out_spk), dim=1)
                    elif self.recurrency_type == 'stacked':
                        layer_in = torch.cat((spk, self.hidden_state[idx+1]), dim=1)
                    elif self.recurrency_type == 'dense':
                        layer_in = torch.cat((inp, *out_spk, spk), dim=1)
                    else:
                        # for dt and none
                        layer_in = spk
                mems[idx] = mem
                losses[idx] = loss
                out_spk.append(spk)

            if self.recurrency_type != 'none':
                if self.recurrency_type == 'dt':
                    self.hidden_state = out_spk[-1]
                else:
                    self.hidden_state = out_spk

        return out_spk, mems, losses

class EchoSpike_layer(nn.Module):
    def __init__(self, num_inputs:int, num_hidden: int, beta:float, n_time_steps: int=100, c_y=[1e-4, -1e-4], inp_thr=0.05):
        """
        Initializes an EchoSpike layer for static data.

        Parameters:
            num_inputs (int): The number of input features.
            num_hidden (int): The number of hidden units.
            beta (float): The leaky parameter for the leaky integrate-and-fire neuron.
            n_time_steps (int): The number of time steps.
        """
        super().__init__()
        self.fc = nn.Linear(num_inputs, num_hidden, bias=False)
        with torch.no_grad():
            # factor 3 necessary for nmnist, because:
            # too small weights create no spikes at all -> no learning
            k = 3/torch.sqrt(torch.tensor(num_inputs))
            self.fc.weight = nn.init.uniform_(self.fc.weight, -k, k)
        if beta < 0: # heterogenous
            mode = torch.log(1/(1+torch.tensor(beta))) # if beta < 0: -beta is the mode of the distribution of betas
            self.beta = mode*torch.ones(num_hidden) + torch.randn(num_hidden)
            self.beta = torch.sigmoid(self.beta)
            print(torch.mean(self.beta), torch.std(self.beta), torch.min(self.beta), torch.max(self.beta))
        else:
            self.beta = beta
        self.lif = snn.Leaky(beta=self.beta) # , reset_mechanism='zero')
        self.n_time_steps = n_time_steps
        self.inp_trace, self.spk_trace = None, None
        self.prev_spk_trace = None
        self.trace_decay = abs(beta)
        self.c_y = c_y.copy()
        self.inp_thr = inp_thr
        self.current_dW = None
        self.acc = torch.zeros(2)
        self.reset()

    def reset(self):
        """Resets the parameters

        Returns:
            torch.tensor: The accuracy of the layer
        """
        self.mem = self.lif.init_leaky()
        acc = torch.zeros(2)
        self.t = 0
        if self.spk_trace is not None:
            acc = torch.ones(2) - self.acc
            self.acc = torch.zeros(2)
            # norm = self.spk_trace.sum(axis=-1).unsqueeze(-1)
            # norm = torch.where(norm > 0, norm, torch.ones_like(norm)) # avoid division by zero
            self.prev_spk_trace = self.spk_trace #/ norm
            self.spk_trace = None
            self.inp_trace = None
        return acc
    
    def loss(self, current):
        if not (self.prev_spk_trace is None or self.spk_trace is None):
            con = (current * self.prev_spk_trace).sum(axis=-1)
            return -(current * self.spk_trace).sum(axis=-1), con
        return torch.zeros(current.shape[0]), torch.zeros(current.shape[0])

    @staticmethod
    def _surrogate(x):
        # The surrogate gradient for the leaky integrate-and-fire neuron
        surr = 1 / (torch.pi * (1 + (torch.pi * x) ** 2))
        return surr

    def _update_trace(self, trace, spk, decay=True):
        if trace is None:
            # initialize trace
            if decay:
                trace = spk
            else:
                trace = spk
        elif decay:
            # decaying trace
            trace = self.trace_decay * trace + spk
        else:
            # non decaying trace
            trace = (trace*self.t + spk) / (self.t + 1)
        return trace
     
    def forward(self, inp, dropin=0, inp_activity=None):
        """Forward pass of the EchoSpike layer

        Args:
            inp (torch.tensor): Spike input to the layer of shape (batch_size, num_inputs)
            bf (int): The broadcasting factor
            dropin (float, optional): Fraction of neurons to radomly activate as a regularization. Defaults to 0.
            inp_activity (torch.tensor, optional): For dynamical online learning. Defaults to None.

        Returns:
            torch.tensor: The spike output of the layer of shape (batch_size, num_hidden)
            torch.tensor: The membrane potential of the neurons (batch_size, num_hidden)
            float: The mean loss of the layer
        """
        inp = torch.atleast_2d(inp)
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        loss_pred, loss_con = self.loss(spk)

        if self.training:
            self.inp_trace = self._update_trace(self.inp_trace, inp)
            if dropin > 0:
                rand_spks = torch.bernoulli(torch.ones_like(spk) * dropin)
                spk = torch.clamp(spk + rand_spks, max=1)
            if self.prev_spk_trace is not None:
                # Online Learning Rule
                if inp_activity is None:
                    raise ValueError('If Model is in training, inp_activity must be provided in forward pass.')
                loss_pred += self.c_y[0]*inp_activity
                loss_con += self.c_y[1]*inp_activity
                dL_in = inp_activity > self.inp_thr
                dL_con = (loss_con > 0) * dL_in
                dL_pred = (loss_pred > 0) * dL_in
                self.acc += torch.tensor([dL_pred.float().mean(), dL_con.float().mean()])/self.n_time_steps
                if self.spk_trace is not None:
                    dW_pred = -torch.einsum('bi, bj->bij', self.spk_trace * EchoSpike_layer._surrogate(self.mem - 1), self.inp_trace)
                    dW_pred = torch.einsum('bvw,b->vw', dW_pred, dL_pred.float())
                else:
                    dW_pred = 0
                dW_con = torch.einsum('bi, bj->bij', self.prev_spk_trace * EchoSpike_layer._surrogate(self.mem - 1), self.inp_trace)
                dW_con = torch.einsum('bvw,b->vw', dW_con, dL_con.float())
                self.fc.weight.grad = dW_con + dW_pred
        self.spk_trace = self._update_trace(self.spk_trace, spk, decay=False)
        self.t += 1
        return spk, self.mem, torch.tensor([loss_pred.mean(), loss_con.mean()])


class simple_out(nn.Module):
    def __init__(self, num_inputs, num_out, beta):
        super().__init__()
        # feed forward part
        self.out_proj = nn.Linear(num_inputs, num_out, bias=False)
        self.lif = snn.Leaky(beta=beta, reset_mechanism='none')
        self.num_out = num_out
        self.dW = None
        self.beta = beta
        self.inp_trace = None
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()
        self.inp_trace = None
        self.dW = None
     
    def _surrogate(self, x):
        return 1 / (torch.pi * (1 + (torch.pi * x) ** 2))
    
    def _update_trace(self, spk):
        if self.inp_trace is None:
            self.inp_trace = spk
        else:
            self.inp_trace = self.beta*self.inp_trace + spk

    def forward(self, inp, target):
        cur = self.out_proj(inp)
        spk, self.mem = self.lif(cur, self.mem)
        if self.training:
            self._update_trace(inp)
            if target is not None:
                # prediction weight update, at last time step of the samples
                target_spk = nn.functional.one_hot(target.long(), num_classes=self.num_out).float()
                pred = nn.Softmax(dim=-1)(self.mem)
                self.out_proj.weight.grad = -torch.einsum('bi, bj -> ij' , (target_spk-pred), self.inp_trace)
                # self.out_proj.weight.grad = -torch.einsum('bi, bj -> ij' , (target_spk-pred) * self._surrogate(self.mem -1), self.inp_trace)

        return spk, self.mem