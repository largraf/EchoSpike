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

class CLAPP_Sequence_SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, num_outputs,
                 beta=0.96, segment_length=20, out_proj=True):
        """
        Initializes the CLAPP SNN with the given parameters.
        This class is used for data with temporal structure, such as SHD.

        Args:
            num_inputs (int): The number of input units.
            num_hidden (list): A list of integers representing the number of hidden units in each layer.
            num_outputs (int): The number of output units.
            beta (float, optional): The beta value for initializing the leaky integrate and fire model. Defaults to 0.75.
            segment_length (int, optional): The number of time bins per segment. Defaults to 20.
            out_proj (bool, optional): Whether to include an output layer. Defaults to True.
        """
        super().__init__()
        self.has_out_proj = out_proj
        layer_type = CLAPP_layer_bio
        # layer_type = CLAPP_layer_segmented
        # Initialized the CLAPP layers with shapes from num_hidden
        self.clapp = torch.nn.ModuleList([layer_type(num_inputs, num_hidden[0], beta)])
        for idx_hidden in range(1, len(num_hidden)):
            self.clapp.append(layer_type(num_hidden[idx_hidden-1],
                                          num_hidden[idx_hidden], beta))

        # initialize output layer
        if self.has_out_proj:
            self.out_proj = CLAPP_out(num_hidden[-1], num_outputs, beta)#nn.Linear(num_hidden[-1], num_outputs)
        self.segment_length = segment_length
    
    def reset(self):
        for clapp_layer in self.clapp:
            clapp_layer.reset()
        if self.has_out_proj:
            self.out_proj.reset()

    def forward(self, inp, target, bf: int, freeze: list=[]):
        with torch.no_grad():
            mems = len(self.clapp)*[None]
            losses = torch.zeros(len(self.clapp))
            clapp_in = inp
            out_spk = []
            for idx, clapp_layer in enumerate(self.clapp):
                factor = bf if not idx in freeze else ''
                clapp_in, mem, loss = clapp_layer(clapp_in, factor)
                mems[idx] = mem
                losses[idx] = loss
                out_spk.append(clapp_in)
            # Final output projection
            self.hidden_state = clapp_in
            if self.has_out_proj:
                clapp_in, out_mem = self.out_proj(clapp_in, target)
                out_spk.append(clapp_in)


        return out_spk, torch.stack(mems), losses

class CLAPP_layer_bio(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta):
        super().__init__()
        # feed forward part
        self.fc = nn.Linear(num_inputs, num_hidden, bias=False)
        with torch.no_grad():
            # too small weights create no spikes at all -> no learning
            self.fc.weight *= 4
        self.lif = snn.Leaky(beta=beta) #), reset_mechanism='zero')
        # Recursive feedback
        self.feedback_trace = None
        self.pred = None
        self.spk_trace = None
        self.inp_trace = None
        self.negative_spk_trace = None
        self.trace_decay = beta
        predict_steps = 20
        self.feedback_trace_delay = deque([torch.zeros(num_hidden) for _ in range(predict_steps)], maxlen=predict_steps)
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()
        self.feedback_trace = None
        if self.spk_trace is not None:
            self.negative_spk_trace = self.spk_trace / 10.
        else:
            self.negative_spk_trace = None
        self.spk_trace = None
        self.inp_trace = None
    
    def CLAPP_loss(self, bf, current, inp):
        if bf == 1:
            # inp.sum() should ensure that the number of spikes doesn't decay with depth
            return torch.relu(inp.sum() - (current * torch.where(self.feedback_trace > 0, self.feedback_trace, -1.)).sum())
        else:
            return torch.relu(-inp.sum()/10 + (current * torch.where(self.negative_spk_trace > 0, self.negative_spk_trace, -1.)).sum())


    @staticmethod
    def _surrogate(x):
        return 1 / (torch.pi * (1 + (torch.pi * x) ** 2))

    def _update_trace(self, trace, spk, no_dec=False):
        if trace is None:
            return spk
        elif not no_dec:
            return self.trace_decay * trace + spk
        return trace + spk
    
    def _update_feedback_trace(self, spk):
        fb = self.feedback_trace_delay[0]
        self.feedback_trace_delay.append(spk)
        if self.feedback_trace is None:
            return fb
        return self.trace_decay * self.feedback_trace + spk
     
    def _dL(self, loss) -> bool:
        return loss > 0

    def forward(self, inp, event):
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)

        loss, loss_contrastive = 0, 0
        self.spk_trace = self._update_trace(self.spk_trace, spk, no_dec=True)
        if self.training:
            self.inp_trace = self._update_trace(self.inp_trace, inp)
            if self.feedback_trace is not None:
                loss = self.CLAPP_loss(1, spk, inp)
                if self.negative_spk_trace is not None:
                    loss_contrastive = self.CLAPP_loss(-1, spk, inp)

            dW = None
            # predictive update
            surr = CLAPP_layer_bio._surrogate(self.mem + cur - 1)
            if self._dL(loss):
                dW = torch.outer(self.feedback_trace * surr, self.inp_trace)
            elif self.feedback_trace is not None:
                self.feedback_trace *= 1
            
            # contrastive update
            if self.negative_spk_trace is not None and self._dL(loss_contrastive):
                if dW is None:
                    dW = -torch.outer(self.negative_spk_trace * surr, self.inp_trace)
                else:
                    dW -= torch.outer(self.negative_spk_trace * surr, self.inp_trace)
            elif self.feedback_trace is not None and spk.sum()>10:
                self.feedback_trace *= 1
            
            
            if dW is not None:
                if self.fc.weight.grad is None:
                    self.fc.weight.grad = - dW
                else:
                    self.fc.weight.grad -= dW

            # update feedback
            self.feedback_trace = self._update_feedback_trace(spk)
        elif self.feedback_trace is not None:
            loss = self.CLAPP_loss(1, self.spk_trace)
        return spk, spk if self.spk_trace is None else self.spk_trace, (loss + loss_contrastive) / 2


class CLAPP_layer_segmented(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta):
        super().__init__()
        # feed forward part
        self.fc = nn.Linear(num_inputs, num_hidden, bias=False)
        with torch.no_grad():
            # too small weights create no spikes at all -> no learning
            self.fc.weight *= 8
        self.lif = snn.Leaky(beta=beta) #, reset_mechanism='zero')
        # Recursive feedback
        self.prediction = None
        self.inp_trace, self.spk_trace = None, None
        self.prev_spk_trace, self.prev_inp_trace = None, None
        self.negative_spk_trace = None
        self.trace_decay = beta
        self.pred = nn.Linear(num_hidden, num_hidden, bias=False)
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()
        if self.spk_trace is not None:
            self.negative_spk_trace = self.spk_trace
            self.spk_trace = None
            self.inp_trace = None
    
    def CLAPP_loss(self, bf, current):
        return bf * torch.square(current - self.prediction).sum()#torch.relu(1 - bf * (current * self.prediction).sum())

    @staticmethod
    def _surrogate(x):
        return 1 / (torch.pi * (1 + (torch.pi * x) ** 2))
        # return torch.where(x < torch.ones(x.shape, device=x.device), 1 / (torch.pi * (1 + (torch.pi * x) ** 2, torch.zeros(x.shape, device=x.device))))

    def _update_trace(self, trace, spk):
        if trace is None:
            trace = spk
        else:
            trace = trace + spk/10# self.trace_decay * trace + spk
        return trace
     
    def _dL(self, loss) -> bool:
        return loss > 0

    def forward(self, inp, event):
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        loss, loss_contrastive = 0, 0
        self.spk_trace = self._update_trace(self.spk_trace, spk)
        if self.training:
            self.inp_trace = self._update_trace(self.inp_trace, inp)
            if 'evaluate' in event:
                if self.prediction is not None:
                    loss = self.CLAPP_loss(1, self.spk_trace)
                    if self.negative_spk_trace is not None:
                        loss_contrastive = self.CLAPP_loss(-1, self.negative_spk_trace)

                # update the weights according to CLAPP learning rule
                # retrodiction = nn.functional.linear(self.spk_trace, self.pred.weight.T)
                # first part Forward weights update
                dW, dW_pred = None, None
                # predictive
                surr = CLAPP_layer_segmented._surrogate(self.mem + cur - 1)
                if loss > 20: #self._dL(loss):
                    dW = torch.outer((self.prediction-self.spk_trace) * surr, self.inp_trace)
                    # dW_pred = torch.outer(self.spk_trace, self.prev_spk_trace)
                # contrastive
                if self.negative_spk_trace is not None and loss_contrastive > -100: #self._dL(loss_contrastive):
                    if dW is None:
                        dW = -torch.outer((self.negative_spk_trace-self.spk_trace) * surr, self.inp_trace)
                    else:
                        dW -= torch.outer((self.negative_spk_trace-self.spk_trace) * surr, self.inp_trace)
                    # if dW_pred is None:
                    #     dW_pred = -torch.outer(self.negative_spk_trace, self.prev_spk_trace)
                    # else:
                    #     dW_pred -= torch.outer(self.negative_spk_trace, self.prev_spk_trace)
                # if dW_pred is not None:
                #     if self.pred.weight.grad is None:
                #         self.pred.weight.grad = - dW_pred
                #     else:
                #         self.pred.weight.grad -= dW_pred
                # second part of forward weight update
                # dW += torch.outer(retrodiction * CLAPP_layer._surrogate(self.prev_spk_trace), self.prev_inp_trace)
                if dW is not None:
                    if self.fc.weight.grad is None:
                        self.fc.weight.grad = - dW
                    else:
                        self.fc.weight.grad -= dW
            if 'predict' in event:
                self.prediction = self.spk_trace#torch.where(self.spk_trace > 0, self.spk_trace, -0.1)
                self.prev_spk_trace = self.spk_trace
                self.prev_inp_trace = self.inp_trace
                self.spk_trace = None
                self.inp_trace = None
        elif event == 'evaluate' and self.prediction is not None:
            loss = self.CLAPP_loss(1, self.spk_trace)
        return spk, self.prev_spk_trace if self.spk_trace == None else self.spk_trace, (loss + loss_contrastive) / 2


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
            self.fc.weight *= 8
        self.lif = snn.Leaky(beta=beta, reset_mechanism='zero')
        self.n_time_steps = n_time_steps
        # Recursive feedback
        self.feedback = None
        self.inp_trace, self.spk_trace = None, None
        self.prev_spk_trace, self.prev_inp_trace = None, None
        self.trace_decay = beta
        self.pred = nn.Linear(num_hidden, num_hidden, bias=False)
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()
        if self.spk_trace is not None:
            self.feedback = torch.where(self.spk_trace > 0, self.spk_trace, -0.1) #self.pred(self.spk_trace)
            self.prev_spk_trace = self.spk_trace
            self.prev_inp_trace = self.inp_trace
            self.spk_trace = None
            self.inp_trace = None
    
    def CLAPP_loss(self, bf, current):
        return torch.relu(1 - bf * (current * self.feedback).sum())

    @staticmethod
    def _surrogate(x):
        # surr = 1 / (torch.pi * (1 + (torch.pi * x) ** 2))
        return torch.heaviside(x, torch.zeros_like(x))

    def _update_trace(self, trace, spk):
        # non decaying trace for static data
        if trace is None:
            trace = spk
        else:
            trace = trace + spk/self.n_time_steps
        return trace
     
    def _dL(self, loss) -> bool:
        return loss > 0

    def forward(self, inp, bf, dropin=0):
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        loss = 0
        self.spk_trace = self._update_trace(self.spk_trace, spk)
        if self.training:
            self.inp_trace = self._update_trace(self.inp_trace, inp)
            if dropin > 0:
                rand_spks = torch.bernoulli(torch.ones_like(spk) * dropin)
                spk = torch.clamp(spk + rand_spks, max=1)
            if self.feedback is not None and bf != 0:
                loss = self.CLAPP_loss(bf, self.spk_trace)
            if bf != 0 and self._dL(loss) and self.prev_spk_trace is not None:
                # update the weights according to CLAPP learning rule
                retrodiction = torch.where(self.spk_trace > 0, self.spk_trace, -0.1)# nn.functional.linear(self.spk_trace, self.pred.weight.T)  #  self.retro(spk)
                # first part Forward weights update
                dW = bf * torch.outer(self.feedback * CLAPP_layer._surrogate(self.spk_trace), self.inp_trace)
                # prediction and retrodiction weight update
                dW_pred = bf * torch.outer(self.spk_trace, self.prev_spk_trace) # (spk, self.prev_spk)
                if self.pred.weight.grad is None:
                    self.pred.weight.grad = - dW_pred
                else:
                    self.pred.weight.grad -= dW_pred
                # second part of forward weight update
                dW += bf * torch.outer(retrodiction * CLAPP_layer._surrogate(self.prev_spk_trace), self.prev_inp_trace)
                if self.fc.weight.grad is None:
                    self.fc.weight.grad = -dW
                else:
                    self.fc.weight.grad -= dW
        elif bf != 0 and self.feedback is not None:
            loss = self.CLAPP_loss(bf, self.spk_trace)
        return spk, self.spk_trace, loss


class CLAPP_out(nn.Module):
    def __init__(self, num_inputs, num_out, beta):
        super().__init__()
        # feed forward part
        self.out_proj = nn.Linear(num_inputs, num_out, bias=False)
        self.lif = snn.Leaky(beta=beta)#, reset_mechanism='zero')
        self.num_out = num_out
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()
    
    def _dL(self, loss):
        return loss > 0

    def _surrogate(self, x):
        return 1 / (torch.pi * (1 + (torch.pi * x) ** 2))

    def forward(self, inp, target):
        cur = self.out_proj(inp)
        spk, self.mem = self.lif(cur, self.mem)
        if self.training:
            # prediction weight update
            target_spk = nn.functional.one_hot(target.long(), num_classes=self.num_out).flatten().float()
            dW = torch.outer((target_spk - spk) * self._surrogate(cur-1), inp)
            if self.out_proj.weight.grad is not None:
                self.out_proj.weight.grad -= dW
            else:
                self.out_proj.weight.grad = -dW

        return spk, self.mem

if __name__ == '__main__': 
    from utils import load_PMNIST, train, test
    train_loader, test_loader = load_PMNIST(4, batch_size=1)

    SNN = CLAPP_SNN(28**2, [512], 10).to('cpu')
    loss_hist, target_list, clapp_loss_hist = train(SNN, train_loader, 1, 'cpu')
    losses, loss_per_digit, clapp_activation, target_list = test(SNN, test_loader, 'cpu')
    print(loss_per_digit)
    print('Accuracy:', losses.sum()/10000)