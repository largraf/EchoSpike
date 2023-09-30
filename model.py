import torch
import torch.nn as nn
import snntorch as snn


# Define Network
class CLAPP_SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden: list, num_outputs, n_time_bins,
                 rule='CLAPP', beta=0.95):
        super().__init__()
        self.n_time_bins = n_time_bins
        self.rule = rule
        self.clapp = [CLAPP_layer(num_inputs, num_hidden[0], beta)]
        for idx_hidden in range(1, len(num_hidden)):
            self.clapp.append(CLAPP_layer(num_hidden[idx_hidden-1],
                                          num_hidden[idx_hidden], beta))

        # initialize layers
        self.fc2 = nn.Linear(num_hidden[-1], num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, inp, bf: int):
        with torch.set_grad_enabled(self.rule == 'backprop' and self.training):
            # only use torch gradients when using backprop
            mem2 = self.lif2.init_leaky()

            spk2_rec = []  # Record the output trace of spikes
            mem2_rec = []  # Record the output trace of membrane potential

            for step in range(self.n_time_bins):
                print(inp[step].shape)
                clapp_in = inp[step].flatten()
                for clapp_layer in self.clapp:
                    clapp_in = clapp_layer(clapp_in, bf)
                # Final output projection
                cur2 = self.fc2(clapp_in)
                spk2, mem2 = self.lif2(cur2, mem2)

                spk2_rec.append(spk2)
                mem2_rec.append(mem2)
                coin_flip = torch.rand(1) > 0.5
                if coin_flip:
                    break
                else:
                    bf = 1

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class CLAPP_layer(nn.Module):
    def __init__(self, num_inputs, num_hidden, beta, lr=2e-6):
        super().__init__()
        self.lr = lr
        # feed forward part
        self.fc = nn.Linear(num_inputs, num_hidden)
        self.lif = snn.Leaky(beta=beta)
        # Recursive feedback
        self.pred = nn.Linear(num_hidden, num_hidden, bias=False)
        self.retro = nn.Linear(num_hidden, num_hidden, bias=False)
        self.feedback = torch.zeros(num_hidden)
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
            self.pred.weight += dW_pred
            self.retro.weight += dW_pred.T
            # Forward weights update
            dW = self.lr * bf * torch.diag(self.feedback) @ torch.outer(surrogate(cur), inp)
            if self.cur_prev is not None:
                dW_c = bf * torch.diag(retrodiction) @ torch.outer(surrogate(self.cur_prev), self.prev_inp)
                dW += dW_c

            self.fc.weight += dW
        # print(self.pred.weight.mean())
        # print(self.fc.weight.mean())
        self.cur_prev = cur
        self.prev_inp = inp
        self.feedback = self.pred(spk)

        return spk

if __name__ == '__main__': 
    from utils import load_NMNIST, train, test
    train_loader, test_loader = load_NMNIST(20, batch_size=1)

    SNN = CLAPP_SNN(2312, [512, 512, 512], 16, 20).to('cpu')
    acc_hist, loss_hist = train(SNN, train_loader, 1, 'cpu')