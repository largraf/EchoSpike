import torch
import torch.nn as nn
import snntorch as snn


class CLAPP_SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, num_outputs,
                 beta=0.75):
        """
        Initializes the CLAPP SNN with the given parameters.

        Args:
            num_inputs (int): The number of input units.
            num_hidden (list): A list of integers representing the number of hidden units in each layer.
            num_outputs (int): The number of output units.
            beta (float, optional): The beta value for initializing the leaky integrate and fire model. Defaults to 0.75.
        """
        super().__init__()
        # Initialized the CLAPP layers with shapes from num_hidden
        self.clapp = torch.nn.ModuleList([CLAPP_layer(num_inputs, num_hidden[0], beta)])
        for idx_hidden in range(1, len(num_hidden)):
            self.clapp.append(CLAPP_layer(num_hidden[idx_hidden-1],
                                          num_hidden[idx_hidden], beta))

        # initialize output layer
        self.out_proj = CLAPP_out(num_hidden[-1], num_outputs, beta)#nn.Linear(num_hidden[-1], num_outputs)
    
    def reset(self):
        for clapp_layer in self.clapp:
            clapp_layer.reset()
        self.out_proj.reset()

    def forward(self, inp, target, bf: int, train_classifier=False):
        with torch.no_grad():
            mems = len(self.clapp)*[None]
            losses = torch.zeros(len(self.clapp))
            clapp_in = inp
            for idx, clapp_layer in enumerate(self.clapp):
                clapp_in, mem, loss = clapp_layer(clapp_in, bf)
                mems[idx] = clapp_in
                losses[idx] = loss
            # Final output projection
            out_spk, out_mem = self.out_proj(clapp_in, target)


        return out_spk, torch.stack(mems), losses


class CLAPP_layer(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta):
        super().__init__()
        # feed forward part
        self.fc = nn.Linear(num_inputs, num_hidden, bias=False)
        with torch.no_grad():
            self.fc.weight *= 10
        self.lif = snn.Leaky(beta=beta, reset_mechanism='zero')
        # Recursive feedback
        self.feedback = None
        self.prev_mem, self.prev_inp, self.prev_spk = None, None, None
        self.pred = nn.Linear(num_hidden, num_hidden, bias=False)
        # self.retro = nn.Linear(num_hidden, num_hidden, bias=False)
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()
    
    def CLAPP_loss(self, bf, cur_spk):
        # cur_spk = cur_spk - 0.5
        return torch.relu(1 - bf * (cur_spk * self.feedback).sum())

    @staticmethod
    def _surrogate(x):
        return 1 / (torch.pi * (1 + (torch.pi * x) ** 2))
        # return torch.where(x < torch.ones(x.shape, device=x.device), 1 / (torch.pi * (1 + (torch.pi * x) ** 2, torch.zeros(x.shape, device=x.device))))

    def _surrogate_feedback(self, bf):
        return (bf * self.feedback < torch.ones_like(self.feedback)).float()
    
    def _dL(self, loss):
        return (loss < 1).float()

    def forward(self, inp, bf, dropin=0):
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        if dropin > 0:
            rand_spks = torch.bernoulli(torch.ones_like(spk) * dropin)
            spk = torch.min(torch.ones_like(spk), spk + rand_spks)
        if self.feedback is not None and bf != 0:
            loss = self.CLAPP_loss(bf, spk)
        else:
            loss = 0
        if self.training and bf != 0:
            # update the weights according to CLAPP learning rule
            bf_2 = bf * self._dL(loss)
            retrodiction = nn.functional.linear(spk, self.pred.weight.T)  #  self.retro(spk)
            # first part Forward weights update
            if self.prev_mem is not None:
                dW = bf_2 * torch.outer(self.feedback * CLAPP_layer._surrogate(self.mem + spk), inp)
                # prediction and retrodiction weight update
                dW_pred = bf_2 * torch.outer(spk, self.prev_spk)
                self.pred.weight.grad = - dW_pred.T
                # second part of forward weight update
                dW = bf_2 * torch.outer(retrodiction * CLAPP_layer._surrogate(self.prev_mem + self.prev_spk), self.prev_inp)

                self.fc.weight.grad = -dW
        # print(self.pred.weight.mean())
        # print(self.fc.weight.mean())
        self.prev_spk = spk
        self.prev_mem = self.mem
        self.prev_inp = inp
        self.feedback = self.pred(spk)

        return spk, self.mem, loss


class CLAPP_out(nn.Module):
    def __init__(self, num_inputs, num_out, beta):
        super().__init__()
        # feed forward part
        self.out_proj = nn.Linear(num_inputs, num_out, bias=False)
        self.lif = snn.Leaky(beta=beta, reset_mechanism='zero')
        self.num_out = num_out
        self.reset()

    def reset(self):
        self.mem = self.lif.init_leaky()

    def forward(self, inp, target):
        cur = self.out_proj(inp)
        spk, self.mem = self.lif(cur, self.mem)
        if self.training:
            # prediction weight update
            target_spk = nn.functional.one_hot(target, num_classes=self.num_out)[0].float()
            dW = torch.outer(target_spk - 0.1, inp)
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