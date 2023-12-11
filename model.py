import torch
import torch.nn as nn
import snntorch as snn
from collections import deque


class CLAPP_SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, num_outputs,
                 beta=0.9, out_proj=True):
        """
        Initializes the CLAPP SNN with the given parameters.
        This class is used for static data, such as poisson encodings of images.

        Args:
            num_inputs (int): The number of input units.
            num_hidden (list): A list of integers representing the number of hidden units in each layer.
            num_outputs (int): The number of output units.
            beta (float, optional): The beta value for initializing the leaky integrate and fire model. Defaults to 0.75.
            out_proj (bool, optional): Whether to include an output layer. Defaults to True.
        """
        super().__init__()
        self.has_out_proj = out_proj
        # Initialized the CLAPP layers with shapes from num_hidden
        self.clapp = torch.nn.ModuleList([CLAPP_layer(num_inputs, num_hidden[0], beta)])
        for idx_hidden in range(1, len(num_hidden)):
            self.clapp.append(CLAPP_layer(num_hidden[idx_hidden-1],
                                          num_hidden[idx_hidden], beta))

        # initialize output layer
        if self.has_out_proj:
            self.out_proj = CLAPP_out(num_hidden[-1], num_outputs, beta)
 
    def reset(self):
        for clapp_layer in self.clapp:
            clapp_layer.reset()
        if self.has_out_proj:
            self.out_proj.reset()

    def forward(self, inp, target, bf: int, freeze: list=[]):
        with torch.no_grad():
            spike_traces = len(self.clapp)*[None]
            out_spk = []
            losses = torch.zeros(len(self.clapp))
            clapp_in = inp
            for idx, clapp_layer in enumerate(self.clapp):
                factor = bf if not idx in freeze else 0
                clapp_in, spk_trace, loss = clapp_layer(clapp_in, factor)
                out_spk.append(clapp_in)
                spike_traces[idx] = spk_trace
                losses[idx] = loss
            # Final output projection
            if self.has_out_proj:
                spk, out_mem = self.out_proj(clapp_in, target)
                out_spk.append(spk)

        return out_spk, torch.stack(spike_traces), losses


class CLAPP_RSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, beta=0.95, n_time_steps=100, recurrency_type='none', device='cuda', cat=False, online=False):
        """
        Initializes the CLAPP SNN with the given parameters.

        Args:
            num_inputs (int): The number of input units.
            num_hidden (list): A list of integers representing the number of hidden units in each layer.
            beta (float, optional): The beta value for initializing the leaky integrate and fire model. Defaults to 0.75.
            n_time_steps (int, optional): The number of time bins per segment. Defaults to 100.
        """
        super().__init__()
        layer_type = CLAPP_layer_temporal
        self.num_hidden = num_hidden
        self.cat = cat
        self.recurrent = recurrency_type != 'none'
        self.recurrency_type = recurrency_type
        self.device = device

        # Initialized the CLAPP layers with shapes from num_hidden
        if self.recurrent:
            if not cat:
                self.clapp = torch.nn.ModuleList([layer_type(num_inputs+num_hidden[0], num_hidden[0], beta, n_time_steps=n_time_steps, online=online)])
            else:
                self.clapp = torch.nn.ModuleList([layer_type(num_inputs+sum(num_hidden), num_hidden[0], beta, n_time_steps=n_time_steps, online=online)])

        else:
            self.clapp = torch.nn.ModuleList([layer_type(num_inputs, num_hidden[0], beta, n_time_steps=n_time_steps, online=online)])
        for idx_hidden in range(1, len(num_hidden)):
            if self.recurrent:
                if not cat:
                    self.clapp.append(layer_type(num_hidden[idx_hidden-1] + num_hidden[idx_hidden], num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online))
                else:
                    self.clapp.append(layer_type(num_inputs + sum(num_hidden), num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online))
            else:
                if not cat:
                    self.clapp.append(layer_type(num_hidden[idx_hidden-1],
                                                    num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online))
                else:
                    self.clapp.append(layer_type(num_inputs + sum(num_hidden[:idx_hidden-1]),
                                                    num_hidden[idx_hidden], beta, n_time_steps=n_time_steps, online=online))

        if self.recurrent:
            self.hidden_state = None    

    def reset(self, bf):
        clapp_accuracies = torch.zeros(len(self.clapp))
        for i, clapp_layer in enumerate(self.clapp):
            clapp_accuracies[i] = clapp_layer.reset(bf)
        if self.recurrent:
            self.hidden_state = None
        return clapp_accuracies


    def forward(self, inp, target, bf: int, freeze: list=[]):
        with torch.no_grad():
            mems = len(self.clapp)*[None]
            losses = torch.zeros(len(self.clapp), device=inp.device)
            if hasattr(self, 'hidden_state'):
                if self.hidden_state is None:
                    self.hidden_state = [torch.zeros(inp.shape[0], self.num_hidden[i], device=inp.device) for i in range(len(self.num_hidden))]
                if not self.cat:
                    clapp_in = torch.cat((inp, self.hidden_state[0]), dim=1)
                else:
                    clapp_in = torch.cat((inp, *self.hidden_state), dim=1)
            else:
                clapp_in = inp
            out_spk = []
            for idx, clapp_layer in enumerate(self.clapp):
                factor = bf if not idx in freeze else 0
                spk, mem, loss = clapp_layer(clapp_in, factor)
                if idx < len(self.clapp) - 1:
                    if self.recurrent:
                        clapp_in = torch.cat((spk, *self.hidden_state[idx+1:], inp, *out_spk), dim=1)
                    else:
                        clapp_in = spk
                mems[idx] = mem
                losses[idx] = loss
                out_spk.append(spk)
            # Final output projection
            if self.recurrent:
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
        # feed forward part
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
        self.mem = self.lif.init_leaky()
        dL = None
        if self.spk_trace is not None:
            if self.sample_loss is not None:
                if bf == 1:
                    self.sample_loss += 1e-2*self.n_time_steps*self.spk_trace.shape[-1]
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
    def _surrogate(x, loss):
        surr = 1 / (torch.pi * (1 + (torch.pi * x) ** 2))
        return surr#return torch.einsum('b,bn->bn', loss > 0, surr)

    def _update_trace(self, trace, spk, decay=True):
        # non decaying trace for static data
        if trace is None:
            if decay:
                trace = spk
            else:
                trace = spk / self.n_time_steps
        elif decay:
            trace = self.trace_decay * trace + spk
        else:
            trace = trace + spk / self.n_time_steps
        return trace
     
    def forward(self, inp, bf, dropin=0):
        inp = torch.atleast_2d(inp)
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        loss = torch.tensor(0.)
        self.spk_trace = self._update_trace(self.spk_trace, spk, decay=False)
        if self.training and bf != 0:
            self.inp_trace = self._update_trace(self.inp_trace, inp)
            if dropin > 0:
                rand_spks = torch.bernoulli(torch.ones_like(spk) * dropin)
                spk = torch.clamp(spk + rand_spks, max=1)
            if self.prev_spk_trace is not None and bf != 0:
                loss = self.CLAPP_loss(bf, spk)
                if self.sample_loss is not None:
                    self.sample_loss += loss
                else:
                    self.sample_loss = loss

                # update the weights according to adaptation from CLAPP learning rule
                if not self.online:
                    if self.current_dW is not None:
                        self.current_dW += bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * CLAPP_layer_temporal._surrogate(self.mem - 1, loss), self.inp_trace)
                    else:
                        self.current_dW = bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * CLAPP_layer_temporal._surrogate(self.mem - 1, loss), self.inp_trace)
                else:
                    if bf == 1:
                        loss += 0.2*self.spk_trace.shape[-1]*inp.mean()
                        dL = loss > 0
                        self.debug_counter[0] += dL.sum()
                        self.debug_counter[1] += loss.sum()
                    else:
                        loss -= 0.1*self.spk_trace.shape[-1]*inp.mean()
                        dL = loss > 0
                        self.debug_counter[2] += dL.sum()
                        self.debug_counter[3] += loss.sum()
                    current_dW = bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * CLAPP_layer_temporal._surrogate(self.mem - 1, loss), self.inp_trace)
                    self.fc.weight.grad = -torch.einsum('bvw,b->vw', current_dW, dL.float())

        elif bf != 0 and self.prev_spk_trace is not None:
            loss = self.CLAPP_loss(bf, spk)
        return spk, self.spk_trace, loss.mean()


class CLAPP_layer(nn.Module):
    def __init__(self, num_inputs:int, num_hidden: int, beta:float, n_time_steps: int=10):
        """
        Initializes a CLAPP layer for static data.

        Parameters:
            num_inputs (int): The number of input features.
            num_hidden (int): The number of hidden units.
            beta (float): The leaky parameter for the leaky integrate-and-fire neuron.
            n_time_steps (int): The number of time steps.
        """
        super().__init__()
        # feed forward part
        self.fc = nn.Linear(num_inputs, num_hidden, bias=False)
        with torch.no_grad():
            # too small weights create no spikes at all -> no learning
            self.fc.weight *= 3
        self.lif = snn.Leaky(beta=beta) # , reset_mechanism='zero')
        self.n_time_steps = n_time_steps
        self.inp_trace, self.spk_trace = None, None
        self.prev_spk_trace, self.prev_inp_trace = None, None
        self.trace_decay = beta
        # self.pred = nn.Linear(num_hidden, num_hidden, bias=False)
        self.debug_counter = [0, 0, 0, 0]
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()
        if self.spk_trace is not None:
            self.prev_spk_trace = self.spk_trace
            self.prev_inp_trace = self.inp_trace
            self.spk_trace = None
            self.inp_trace = None
    
    def CLAPP_loss(self, bf, current):
        fb = self.prev_spk_trace - self.prev_spk_trace.mean(axis=-1).unsqueeze(-1)
        if bf == 1:
            return torch.relu(0.01*fb.shape[-1] - (current * fb).sum(axis=-1))
        else:
            return torch.relu((current * fb).sum(axis=-1))

    @staticmethod
    def _surrogate(x, loss):
        surr = torch.heaviside(x, torch.zeros_like(x))
        return torch.einsum('b,bn->bn', loss > 0, surr)

    def _update_trace(self, trace, spk):
        # non decaying trace for static data
        if trace is None:
            trace = spk
        else:
            trace = trace + spk/self.n_time_steps
        return trace
     
    def forward(self, inp, bf, dropin=0):
        batched = inp.ndim == 2
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        loss = torch.tensor(0.)
        self.spk_trace = self._update_trace(self.spk_trace, spk)
        if self.training:
            self.inp_trace = self._update_trace(self.inp_trace, inp)
            if dropin > 0:
                rand_spks = torch.bernoulli(torch.ones_like(spk) * dropin)
                spk = torch.clamp(spk + rand_spks, max=1)
            if self.prev_spk_trace is not None and bf != 0:
                loss = self.CLAPP_loss(bf, self.spk_trace)
                if bf == 1:
                    self.debug_counter[0] += (loss > 0).sum()
                    self.debug_counter[1] += loss.mean()
                else:
                    self.debug_counter[2] += (loss > 0).sum()
                    self.debug_counter[3] += loss.mean()

                # update the weights according to adaptation from CLAPP learning rule
                # first part Forward weights update
                if batched:
                    dW = bf * torch.einsum('bi, bj->bij', self.prev_spk_trace * CLAPP_layer._surrogate(self.spk_trace, loss), self.inp_trace).mean(axis=0)
                    dW += bf * torch.einsum('bi, bj->bij', self.spk_trace * CLAPP_layer._surrogate(self.prev_spk_trace, loss), self.prev_inp_trace).mean(axis=0)
                else:
                    dW = bf * torch.einsum('i, j->ij', self.prev_spk_trace * CLAPP_layer._surrogate(self.spk_trace, loss), self.inp_trace)
                    dW += bf * torch.einsum('i, j->ij', self.spk_trace * CLAPP_layer._surrogate(self.prev_spk_trace, loss), self.prev_inp_trace)
                # prediction and retrodiction weight update
                # dW_pred = bf * torch.outer(self.spk_trace, self.prev_spk_trace) 
                # if self.pred.weight.grad is None:
                #     self.pred.weight.grad = - dW_pred
                # else:
                #     self.pred.weight.grad -= dW_pred
                # second part of forward weight update
                if self.fc.weight.grad is None:
                    self.fc.weight.grad = -dW
                else:
                    self.fc.weight.grad -= dW
        elif bf != 0 and self.prev_spk_trace is not None:
            loss = self.CLAPP_loss(bf, self.spk_trace)
        return spk, self.spk_trace, loss.mean()


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