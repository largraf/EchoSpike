from utils import load_NMNIST, load_half_MNIST, train, test
from model import CLAPP_SNN
import numpy as np
import torch

device = 'cpu'
epochs = 3
batch_size = 1
n_inputs = 28*14 #34 * 34 * 2
n_hidden = 2 * [512]
n_outputs = 10
model_name = 'SNN_CLAPP_0_patches.pt'

# load dataset
n_time_bins = 4
train_loader, test_loader = load_half_MNIST() #load_NMNIST(n_time_bins, batch_size=batch_size)

# train and save model
SNN = CLAPP_SNN(n_inputs, n_hidden, n_outputs).to(device)
loss_hist, target_list, clapp_loss_hist = train(SNN, train_loader, epochs, device)
torch.save(SNN.state_dict(), f'models/{model_name}.pt')

# test model
losses, loss_per_digit, clapp_activation, target_list, clapp_losses = test(SNN, test_loader, device)
print(loss_per_digit)
print('Mean Loss:', losses.sum()/10000)



