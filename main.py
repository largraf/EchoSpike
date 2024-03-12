from utils import train
from data import load_classwise_PMNIST, load_classwise_NMNIST, load_SHD
from model import EchoSpike
import torch
import pickle

# hyperparameters
class Args:
    def __init__(self):
        self.model_name = 'test'
        self.dataset = 'shd'
        self.online = True
        self.device = 'cpu'
        self.recurrency_type = 'dense'
        self.lr = 1e-4
        self.epochs = 1000
        self.augment = True
        self.batch_size = 128 # 64 saccade and 64 predictive before weight update -> 128
        self.n_hidden = 4*[450]
        if self.dataset == 'nmnist':
            self.c_y = [1e-4, -1e-4] if not self.online else [2, -1]
            self.inp_thr = 0.02
            self.n_inputs = 2*34*34
            self.n_outputs = 10
            self.n_time_bins = 10
            self.beta = 0.9
        elif self.dataset == 'pmnist':
            self.c_y = [1e-4, -1e-4]
            self.inp_thr = 0.0
            self.n_inputs = 28*28
            self.n_outputs = 10
            self.poisson_scale = 0.8
            self.n_time_bins = 10
            self.beta = 0.9
        elif self.dataset == 'shd':
            self.inp_thr = 0.05
            self.c_y = [8e-4, -4e-4] if not self.online else [1.5, -1.5]
            self.n_inputs = 700
            self.n_outputs = 20
            self.n_time_bins = 100
            self.beta = [0.94,0.96,0.98,1.0]#-0.95
        else:
            raise NotImplementedError

if __name__ == '__main__':
    args = Args()
    torch.manual_seed(123)
    # load dataset
    if args.dataset == 'nmnist':
        train_loader, _, test_loader = load_classwise_NMNIST(args.n_time_bins, split_train=True, batch_size=args.batch_size) 
    elif args.dataset == 'pmnist':
        train_loader, test_loader = load_classwise_PMNIST(args.n_time_bins, scale=args.poisson_scale, batch_size=args.batch_size)
    elif args.dataset == 'shd':
        train_loader, test_loader = load_SHD(batch_size=args.batch_size)

    # train model
    SNN = EchoSpike(args.n_inputs, args.n_hidden, c_y= args.c_y, beta=args.beta,
                     device=args.device, recurrency_type=args.recurrency_type,
                     n_time_steps=args.n_time_bins, online=args.online, inp_thr=args.inp_thr).to(args.device)

    loss_hist = train(SNN, train_loader, args.epochs, args.device, args.model_name,
                            batch_size=args.batch_size, online=args.online, lr=args.lr, augment=args.augment)

    # Save the model, loss history and arguments
    torch.save(SNN.state_dict(), f'models/{args.model_name}.pt')
    torch.save(loss_hist, f'models/{args.model_name}_loss_hist.pt')
    with open(f'models/{args.model_name}_args.pkl', 'wb') as f:
        pickle.dump(args, f)


