import torch
import torch.nn as nn
import snntorch as snn


class CLAPP_RSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, beta=0.95, n_time_steps=100, recurrency_type='none', device='cuda', online=False):
        """
        Initializes the CLAPP SNN with the given parameters.

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

        # Initialized the CLAPP layers with shapes from num_hidden
        if self.recurrency_type == 'none':
            self.clapp = nn.ModuleList([CLAPP_layer_temporal(num_inputs, num_hidden[0], beta, n_time_steps=n_time_steps, online=online)])
            for idx_hidden in range(1, len(num_hidden)):
                self.clapp.append(CLAPP_layer_temporal(num_hidden[idx_hidden-1],
                                                num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online))
        elif self.recurrency_type == 'stacked':
            self.clapp = nn.ModuleList([CLAPP_layer_temporal(num_inputs+num_hidden[0], num_hidden[0], beta, n_time_steps=n_time_steps, online=online)])
            for idx_hidden in range(1, len(num_hidden)):
                self.clapp.append(CLAPP_layer_temporal(num_hidden[idx_hidden-1] + num_hidden[idx_hidden], num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online))

        elif self.recurrency_type == 'full':
            self.clapp = nn.ModuleList([CLAPP_layer_temporal(num_inputs+sum(num_hidden), num_hidden[0], beta, n_time_steps=n_time_steps, online=online)])
            for idx_hidden in range(1, len(num_hidden)):
                self.clapp.append(CLAPP_layer_temporal(num_inputs + sum(num_hidden), num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online))
        else:
            raise NotImplementedError

        self.hidden_state = None    

    def reset(self, bf):
        clapp_accuracies = torch.zeros(len(self.clapp))
        for i, clapp_layer in enumerate(self.clapp):
            clapp_accuracies[i] = clapp_layer.reset(bf)
        self.hidden_state = None
        return clapp_accuracies

    def forward(self, inp, bf: int, freeze: list=[], inp_activity=None):
        with torch.no_grad():
            mems = len(self.clapp)*[None]
            losses = torch.zeros(len(self.clapp), device=inp.device)
            # define input for first layer
            if self.recurrency_type != 'none':
                if self.hidden_state is None:
                    # init hidden state
                    self.hidden_state = [torch.zeros(inp.shape[0], self.num_hidden[i], device=inp.device) for i in range(len(self.num_hidden))]
                if self.recurrency_type == 'stacked':
                    clapp_in = torch.cat((inp, self.hidden_state[0]), dim=1)
                elif self.recurrency_type == 'full':
                    clapp_in = torch.cat((inp, *self.hidden_state), dim=1)
            else:
                clapp_in = inp
            out_spk = []

            for idx, clapp_layer in enumerate(self.clapp):
                factor = bf if not idx in freeze else 0
                spk, mem, loss = clapp_layer(clapp_in, factor, inp_activity=inp_activity)
                if idx < len(self.clapp) - 1:
                    if self.recurrency_type == 'full':
                        clapp_in = torch.cat((inp, spk, *self.hidden_state[idx+1:], *out_spk), dim=1)
                    elif self.recurrency_type == 'stacked':
                        clapp_in = torch.cat((spk, self.hidden_state[idx+1]), dim=1)
                    else:
                        clapp_in = spk
                mems[idx] = mem
                losses[idx] = loss
                out_spk.append(spk)

            if self.recurrency_type != 'none':
                self.hidden_state = out_spk

        return out_spk, mems, losses

class CLAPP_layer_temporal(nn.Module):
    def __init__(self, num_inputs:int, num_hidden: int, beta:float, n_time_steps: int=100, online=False):
        """
        Initializes a CLAPP layer for static data.

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
                    self.sample_loss += 5e-3*self.n_time_steps*self.spk_trace.shape[-1]
                    dL = (self.sample_loss > 0).float()
                else:
                    self.sample_loss -= 5e-3*self.n_time_steps*self.spk_trace.shape[-1]
                    dL = (self.sample_loss > 0).float()
                if not self.online:
                    self.current_dW = torch.einsum('bvw,b->vw', self.current_dW, dL)
                    if self.fc.weight.grad is None:
                        self.fc.weight.grad = -self.current_dW
                    else:
                        self.fc.weight.grad -= self.current_dW

            self.current_dW = None
            self.sample_loss = None
            self.prev_spk_trace = self.spk_trace
            self.spk_trace = None
            self.inp_trace = None
        if dL is not None:
            return 1 - dL.sum()/len(dL)
        else: return 0
    
    def CLAPP_loss(self, bf, current):
        fb = self.prev_spk_trace - self.prev_spk_trace.mean(axis=-1).unsqueeze(-1)
        if bf == 1:
            return -(current * fb).sum(axis=-1)
        else:
            return (current * fb).sum(axis=-1)

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
        """Forward pass of the CLAPP layer

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
        # Forward pass of the CLAPP layer
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
                loss = self.CLAPP_loss(bf, spk)
                if self.sample_loss is not None:
                    self.sample_loss += loss
                else:
                    self.sample_loss = loss

                # update the weights according to adaptation from CLAPP learning rule
                if not self.online:
                    if self.current_dW is not None:
                        self.current_dW += bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * CLAPP_layer_temporal._surrogate(self.mem - 1), self.inp_trace)
                    else:
                        self.current_dW = bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * CLAPP_layer_temporal._surrogate(self.mem - 1), self.inp_trace)
                else:
                    # Online Learning Rule
                    online_loss = -bf * (spk * self.prev_spk_trace).sum(axis=-1)
                    factor = 2 if bf == 1 else 1 # large factor makes predictive harder and contrastive easier leading to less sparse activity
                    idx = 0 if bf == 1 else 2
                    if inp_activity is None:
                        inp_activity = inp.mean(axis=-1)
                    online_loss += factor*bf*self.prev_spk_trace.mean(axis=-1)*self.spk_trace.shape[-1]*inp_activity
                    dL = (online_loss > 0) * (inp_activity > 0.05)
                    current_dW = bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * CLAPP_layer_temporal._surrogate(self.mem - 1), self.inp_trace)
                    self.fc.weight.grad = -torch.einsum('bvw,b->vw', current_dW, dL.float())
                    self.debug_counter[idx] += dL.sum()
                    self.debug_counter[idx+1] += self.fc.weight.grad.mean()

        elif bf != 0 and self.prev_spk_trace is not None:
            loss = self.CLAPP_loss(bf, spk)
        else:
            loss = torch.tensor(0.)
        return spk, self.mem, loss.mean()


class CLAPP_out(nn.Module):
    def __init__(self, num_inputs, num_out, beta):
        super().__init__()
        # feed forward part
        self.out_proj = nn.Linear(num_inputs, num_out, bias=False)
        self.lif = snn.Leaky(beta=beta)#, reset_mechanism='zero')
        self.num_out = num_out
        self.dW = None
        self.beta = beta
        self.inp_trace = None
        self.reset()

    def reset(self, dL=None):
        self.mem = self.lif.init_leaky()
        self.inp_trace = None
        if self.dW is not None and dL is not None:
            dW = torch.einsum('bvw,b->vw', self.dW, dL)
            self.out_proj.weight.grad = -dW
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
            # prediction weight update
            target_spk = nn.functional.one_hot(target.long(), num_classes=self.num_out).float()
            if self.dW is None:
                self.dW = torch.einsum('bi, bj -> bij' , (target_spk-spk) * self._surrogate(self.mem -1), self.inp_trace)
            else:
                self.dW += torch.einsum('bi, bj -> bij' , (target_spk-spk) * self._surrogate(self.mem -1), self.inp_trace)

        return spk, self.mem

class ETLP_out(nn.Module):
    def __init__(self, num_inputs, num_out, beta):
        super().__init__()
        # feed forward part
        self.out_proj = nn.Linear(num_inputs, num_out, bias=False)
        self.lif = snn.Leaky(beta=beta)#, reset_mechanism='zero')
        self.num_out = num_out
        self.dW = None
        self.beta = beta
        self.inp_trace = None
        self.reset()

    def reset(self, dL=None):
        self.mem = self.lif.init_leaky()
        self.inp_trace = None
        if self.dW is not None and dL is not None:
            dW = torch.einsum('bvw,b->vw', self.dW, dL)
            self.out_proj.weight.grad = -dW
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
            # prediction weight update
            target_spk = nn.functional.one_hot(target.long(), num_classes=self.num_out).float()
            if self.dW is None:
                self.dW = torch.einsum('bi, bj -> bij' , (target_spk-spk) * self._surrogate(self.mem -1), self.inp_trace)
            else:
                self.dW += torch.einsum('bi, bj -> bij' , (target_spk-spk) * self._surrogate(self.mem -1), self.inp_trace)

        return spk, self.mem


if __name__ == '__main__': 
    from utils import train
    from data import load_SHD
    # train_loader, test_loader = load_classwise_SHD()

    SNN = ETLP_out(700, 20, 0.99)
    loss_hist, target_list, clapp_loss_hist = train(SNN, train_loader, 3, 'cpu', 'test_mixer', batch_size=60)
