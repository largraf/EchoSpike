from utils import train_samplewise_clapp, test_SHD
from data import load_classwise_PMNIST
from model import CLAPP_SNN
import numpy as np
import torch
import pickle

# hyperparameters
class Args:
    def __init__(self):
        self.device = 'cpu'
        self.epochs = 3
        self.batch_size = 32
        self.n_inputs = 28*28
        self.n_hidden = 3 * [512]
        self.n_outputs = 10
        self.beta = 0.9
        self.poisson_scale = 0.4
        self.n_time_bins = 10
        self.model_name = 'mnist_3layer_balanced'
if __name__ == '__main__':
    args = Args()
    torch.manual_seed(123)
    # load dataset
    train_loader, test_loader = load_classwise_PMNIST(args.n_time_bins, scale=args.poisson_scale) 

    # train and save model
    SNN = CLAPP_SNN(args.n_inputs, args.n_hidden, args.n_outputs, beta=args.beta, out_proj=False).to(args.device)
    clapp_loss_hist = train_samplewise_clapp(SNN, train_loader, args.epochs, args.device, args.model_name, batch_size=args.batch_size)

    torch.save(SNN.state_dict(), f'models/{args.model_name}.pt')
    torch.save(clapp_loss_hist, f'models/{args.model_name}_clapp_loss_hist.pt')
    with open(f'models/{args.model_name}_args.pkl', 'wb') as f:
        pickle.dump(args, f)


