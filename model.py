import torch
import torch.nn as nn
import snntorch as snn


# Define Network
class CLAPP_SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, num_outputs, n_time_bins,
                 beta=0.75):
        """
        Initializes the CLAPP SNN with the given parameters.

        Args:
            num_inputs (int): The number of input units.
            num_hidden (list): A list of integers representing the number of hidden units in each layer.
            num_outputs (int): The number of output units.
            n_time_bins (int): The number of time bins for the NMNIST dataset.
            beta (float, optional): The beta value for initializing the leaky integrate and fire model. Defaults to 0.75.
        """
        super().__init__()
        self.n_time_bins = n_time_bins
        # Initialized the CLAPP layers with shapes from num_hidden
        self.clapp = torch.nn.ModuleList([CLAPP_layer(num_inputs, num_hidden[0], beta)])
        for idx_hidden in range(1, len(num_hidden)):
            self.clapp.append(CLAPP_layer(num_hidden[idx_hidden-1],
                                          num_hidden[idx_hidden], beta))

        # initialize output layer
        self.out_proj = CLAPP_out(num_hidden[-1], num_outputs, beta)#nn.Linear(num_hidden[-1], num_outputs)

    def forward(self, inp, target, bf: int, train_classifier=False):
        with torch.no_grad():
            spk2_rec = []  # Record the output trace of spikes
            out_spks = []
            if train_classifier:
                for clapp in self.clapp:
                    clapp.eval()
            mem_his = [[] for _ in range(len(self.clapp))]
            for step in range(self.n_time_bins):
                clapp_in = inp[step].flatten()
                for idx, clapp_layer in enumerate(self.clapp):
                    clapp_in, mem = clapp_layer(clapp_in, bf)
                    mem_his[idx].append(mem)
                # Final output projection
                out_spk, out_mem = self.out_proj(clapp_in, target, bf)
                out_spks.append(out_spk)

                coin_flip = torch.rand(1) > 0.5
                if coin_flip and self.training:
                    break
                else:
                    bf = 1

        return torch.stack(out_spks), torch.stack([torch.stack(li) for li in mem_his]).swapaxes(0,1)


class CLAPP_layer(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta, lr=2e-6):
        super().__init__()
        self.lr = lr
        # feed forward part
        self.fc = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif = snn.Leaky(beta=beta)
        # Recursive feedback
        self.pred = nn.Linear(num_hidden, num_hidden, bias=False)
        self.retro = nn.Linear(num_hidden, num_hidden, bias=False)
        self.feedback = torch.zeros(num_hidden)
        self.reset()

    def reset(self):
        self.feedback = torch.zeros(self.feedback.shape[0])
        self.cur_prev, self.prev_inp = None, None
        self.mem = self.lif.init_leaky()

    def forward(self, inp, bf):
        def surrogate(x):
            return 1 / (torch.pi * (1 + (torch.pi * x) ** 2))
        cur = self.fc(inp)
        spk, self.mem = self.lif(cur, self.mem)
        if self.training:
            # update the weights according to CLAPP learning rule
            retrodiction = self.retro(spk)
            # prediction and retrodiction weight update
            dW_pred = self.lr * bf * torch.outer(self.feedback, spk)
            self.pred.weight.grad = dW_pred
            self.retro.weight.grad = dW_pred.T
            # Forward weights update
            dW = self.lr * bf * torch.diag(self.feedback) @ torch.outer(surrogate(cur), inp)
            if self.cur_prev is not None:
                dW_c = bf * torch.diag(retrodiction) @ torch.outer(surrogate(self.cur_prev), self.prev_inp)
                dW += dW_c

            self.fc.weight.grad = dW
        # print(self.pred.weight.mean())
        # print(self.fc.weight.mean())
        self.cur_prev = cur
        self.prev_inp = inp
        self.feedback = self.pred(spk)

        return spk, self.mem


class CLAPP_out(nn.Module):
    def __init__(self, num_inputs, num_out, beta, lr=2e-6):
        super().__init__()
        self.lr = lr
        # feed forward part
        self.out_proj = nn.Linear(num_inputs, num_out, bias=False)
        self.lif = snn.Leaky(beta=beta)
        self.num_out = num_out
        self.reset()

    def reset(self):
        self.feedback = torch.zeros(self.num_out)
        self.cur_prev, self.prev_inp = None, None
        self.mem = self.lif.init_leaky()

    def forward(self, inp, target, bf):
        cur = self.out_proj(inp)
        spk, self.mem = self.lif(cur, self.mem)
        if self.training:
            # prediction weight update
            dW_pred = self.lr * bf * torch.outer(self.feedback, inp)
            self.out_proj.weight.grad = dW_pred
        
        self.feedback = nn.functional.one_hot(target, num_classes=self.num_out)[0].float()

        return spk, self.mem

if __name__ == '__main__': 
    from utils import load_NMNIST, train, test
    train_loader, test_loader = load_NMNIST(20, batch_size=1)

    SNN = CLAPP_SNN(2312, [512, 512, 512], 16, 20).to('cpu')
    loss_hist = train(SNN, train_loader, 1, 'cpu')
    test(SNN, test_loader, 'cpu')