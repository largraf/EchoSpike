from utils import train
from data import load_classwise_PMNIST, load_classwise_NMNIST, load_SHD
from model import CLAPP_RSNN
import torch
import pickle

# hyperparameters
class Args:
    def __init__(self):
        self.model_name = 'nmnist_3layer_allclapp'
        self.dataset = 'nmnist'
        self.online = False
        self.device = 'cpu'
        self.recurrency_type = 'none'
        self.lr = 1e-4
        self.epochs = 3
        self.batch_size = 64
        self.n_hidden = 3*[512]
        self.beta = 0.9
        if self.dataset == 'nmnist':
            self.n_inputs = 2*34*34 #28*28 # 700
            self.n_outputs = 10
            self.n_time_bins = 10
        elif self.dataset == 'pmnist':
            self.n_inputs = 28*28
            self.n_outputs = 10
            self.poisson_scale = 0.8
            self.n_time_bins = 10
        elif self.dataset == 'shd':
            self.n_inputs = 700
            self.n_outputs = 20
            self.n_time_bins = 100

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
    SNN = CLAPP_RSNN(args.n_inputs, args.n_hidden, beta=args.beta,
                     device=args.device, recurrency_type=args.recurrency_type,
                     n_time_steps=args.n_time_bins, online=args.online).to(args.device)

    clapp_loss_hist = train(SNN, train_loader, args.epochs, args.device, args.model_name,
                            batch_size=args.batch_size, online=args.online, lr=args.lr)

    # Save the model, loss history and arguments
    torch.save(SNN.state_dict(), f'models/{args.model_name}.pt')
    torch.save(clapp_loss_hist, f'models/{args.model_name}_clapp_loss_hist.pt')
    with open(f'models/{args.model_name}_args.pkl', 'wb') as f:
        pickle.dump(args, f)


