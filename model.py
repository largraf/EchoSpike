import torch
import torch.nn as nn
import snntorch as snn


class EchoSpike(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, c_y=[1e-4, -1e-4], beta=0.95, n_time_steps=100, recurrency_type='none', device='cuda', online=False):
        """
        Initializes the EchoSpike SNN with the given parameters.

        Args:
            num_inputs (int): The number of input units.
            num_hidden (list): A list of integers representing the number of hidden units in each layer.
            beta (float, optional): The beta value for initializing the leaky integrate and fire model. Defaults to 0.75.
            n_time_steps (int, optional): The number of time bins per segment. Defaults to 100.
            recurrency_type (string, optional): Select the type of recurrency: (none, stacked, full)
            device (string, optional): torch device
            online (bool, optional): Wether the online algorithm is used or not
        """
        super().__init__()
        self.num_hidden = num_hidden
        self.recurrency_type = recurrency_type
        self.device = device

        # Initialized the EchoSpike layers with shapes from num_hidden
        if self.recurrency_type == 'none':
            self.layers = nn.ModuleList([EchoSpike_layer(num_inputs, num_hidden[0], beta, n_time_steps=n_time_steps, online=online, c_y=c_y)])
            for idx_hidden in range(1, len(num_hidden)):
                self.layers.append(EchoSpike_layer(num_hidden[idx_hidden-1],
                                                num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online, c_y=c_y))
        elif self.recurrency_type == 'stacked':
            self.layers = nn.ModuleList([EchoSpike_layer(num_inputs+num_hidden[0], num_hidden[0], beta, n_time_steps=n_time_steps, online=online, c_y=c_y)])
            for idx_hidden in range(1, len(num_hidden)):
                self.layers.append(EchoSpike_layer(num_hidden[idx_hidden-1] + num_hidden[idx_hidden], num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online, c_y=c_y))

        elif self.recurrency_type == 'full':
            self.layers = nn.ModuleList([EchoSpike_layer(num_inputs+sum(num_hidden), num_hidden[0], beta, n_time_steps=n_time_steps, online=online, c_y=c_y)])
            for idx_hidden in range(1, len(num_hidden)):
                self.layers.append(EchoSpike_layer(num_inputs + sum(num_hidden), num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online, c_y=c_y))
        elif self.recurrency_type == 'dt':
            # deep transition RNN: feed back last layer to first layer
            self.layers = nn.ModuleList([EchoSpike_layer(num_inputs+num_hidden[-1], num_hidden[0], beta, n_time_steps=n_time_steps, online=online, c_y=c_y)])
            for idx_hidden in range(1, len(num_hidden)):
                self.layers.append(EchoSpike_layer(num_hidden[idx_hidden-1],
                                                num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online, c_y=c_y))
        else:
            raise NotImplementedError

        self.hidden_state = None    

    def reset(self, bf):
        accuracies = torch.zeros(len(self.layers))
        for i, layer in enumerate(self.layers):
            accuracies[i] = layer.reset(bf)
        self.hidden_state = None
        return accuracies

    def forward(self, inp, bf: int, freeze: list=[], inp_activity=None):
        with torch.no_grad():
            mems = len(self.layers)*[None]
            losses = torch.zeros(len(self.layers), device=inp.device)
            # define input for first layer
            if self.recurrency_type != 'none':
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
                spk, mem, loss = layer(layer_in, factor, inp_activity=inp_activity)
                if idx < len(self.layers) - 1:
                    if self.recurrency_type == 'full':
                        layer_in = torch.cat((inp, spk, *self.hidden_state[idx+1:], *out_spk), dim=1)
                    elif self.recurrency_type == 'stacked':
                        layer_in = torch.cat((spk, self.hidden_state[idx+1]), dim=1)
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
    def __init__(self, num_inputs:int, num_hidden: int, beta:float, n_time_steps: int=100, online=False, c_y=[1e-4, -1e-4]):
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
        self.lif = snn.Leaky(beta=beta) # , reset_mechanism='zero')
        self.beta = beta
        self.n_time_steps = n_time_steps
        self.online = online
        self.inp_trace, self.spk_trace = None, None
        self.prev_spk_trace = None
        self.trace_decay = beta
        self.c_y = c_y
        # self.pred = nn.Linear(num_hidden, num_hidden, bias=False)
        self.debug_counter = [0, 0, 0, 0]
        self.current_dW = None
        self.sample_loss = None
        self.reset(0)

    def reset(self, bf):
        """Resets the parameters and updates the weights (if not online)

        Args:
            bf (int): broadcasting factor 

        Returns:
            torch.tensor: The accuracy of the layer
        """
        self.mem = self.lif.init_leaky()
        dL = None
        if self.spk_trace is not None:
            if self.sample_loss is not None:
                # perform a weight update
                if bf == 1:
                    self.sample_loss += self.c_y[0] * self.n_time_steps * self.spk_trace.shape[-1] # 1e-2 and 5e-3 
                    dL = (self.sample_loss > 0).float()
                else:
                    self.sample_loss += self.c_y[1] * self.n_time_steps * self.spk_trace.shape[-1]
                    dL = (self.sample_loss > 0).float()
                if not self.online:
                    self.current_dW = torch.einsum('bvw,b->vw', self.current_dW, dL)
                    if self.fc.weight.grad is None:
                        self.fc.weight.grad = -self.current_dW
                    else:
                        self.fc.weight.grad -= self.current_dW

            self.current_dW = None
            self.sample_loss = None
            norm = self.spk_trace.sum(axis=-1).unsqueeze(-1)
            norm = torch.where(norm > 0, norm, torch.ones_like(norm)) # avoid division by zero
            self.prev_spk_trace = self.spk_trace / norm
            self.spk_trace = None
            self.inp_trace = None
        if dL is not None:
            return 1 - dL.mean()
        else: return 0
    
    def loss(self, bf, current):
        # fb = self.prev_spk_trace - self.prev_spk_trace.mean(axis=-1).unsqueeze(-1)
        if bf == 1:
            return -(current * self.prev_spk_trace).sum(axis=-1)
        else:
            return (current * self.prev_spk_trace).sum(axis=-1)

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
                trace = spk / self.n_time_steps
        elif decay:
            # decaying trace
            trace = self.trace_decay * trace + spk
        else:
            # non decaying trace
            trace = trace + spk / self.n_time_steps
        return trace
     
    def forward(self, inp, bf, dropin=0, inp_activity=None):
        """Forward pass of the EchoSpike layer

        Args:
            inp (torch.tensor): Spike input to the layer of shape (batch_size, num_inputs)
            bf (int): The broadcasting factor
            dropin (float, optional): Fraction of neurons to radomly activate as a regularization. Defaults to 0.
            inp_activity (torch.tensor, optional): For dynamic online learning. Defaults to None.

        Returns:
            torch.tensor: The spike output of the layer of shape (batch_size, num_hidden)
            torch.tensor: The membrane potential of the neurons (batch_size, num_hidden)
            float: The mean loss of the layer
        """
        inp = torch.atleast_2d(inp)
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        self.spk_trace = self._update_trace(self.spk_trace, spk, decay=False)

        if self.training and bf != 0:
            self.inp_trace = self._update_trace(self.inp_trace, inp)
            if dropin > 0:
                rand_spks = torch.bernoulli(torch.ones_like(spk) * dropin)
                spk = torch.clamp(spk + rand_spks, max=1)
            if self.prev_spk_trace is not None:
                loss = self.loss(bf, spk)
                if self.sample_loss is not None:
                    self.sample_loss += loss
                else:
                    self.sample_loss = loss

                # update the weights according to adaptation from CLAPP/ EchoSpike learning rule
                if not self.online:
                    if self.current_dW is not None:
                        self.current_dW += bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * EchoSpike_layer._surrogate(self.mem - 1), self.inp_trace)
                    else:
                        self.current_dW = bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * EchoSpike_layer._surrogate(self.mem - 1), self.inp_trace)
                else:
                    # Online Learning Rule
                    online_loss = -bf * (spk * self.prev_spk_trace).sum(axis=-1)
                    factor = 2 if bf == 1 else 1 # large factor makes predictive harder and contrastive easier leading to less sparse activity
                    idx = 0 if bf == 1 else 2
                    if inp_activity is None:
                        inp_activity = inp.mean(axis=-1)
                    online_loss += factor*bf*self.prev_spk_trace.mean(axis=-1)*self.spk_trace.shape[-1]*inp_activity
                    dL = (online_loss > 0) * (inp_activity > 0.05)
                    current_dW = bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * EchoSpike_layer._surrogate(self.mem - 1), self.inp_trace)
                    self.fc.weight.grad = -torch.einsum('bvw,b->vw', current_dW, dL.float())
                    self.debug_counter[idx] += dL.sum()
                    self.debug_counter[idx+1] += self.fc.weight.grad.mean()

        elif bf != 0 and self.prev_spk_trace is not None:
            loss = self.loss(bf, spk)
        else:
            loss = torch.tensor(0.)
        return spk, self.mem, loss.mean()


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