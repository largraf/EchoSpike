from utils import load_PMNIST, load_SHD, train_sample_wise, test_old
from model import CLAPP_SNN
import numpy as np
import torch

device = 'cpu'
epochs = 1
batch_size = 1
n_inputs = 700 # 28*28
n_hidden = 2 * [512]
n_outputs = 20
model_name = 'SNN_CLAPP_test'

# load dataset
n_time_bins = 100
train_loader, test_loader = load_SHD(n_time_bins)#, scale=0.4) #load_NMNIST(n_time_bins, batch_size=batch_size)

# train and save model
SNN = CLAPP_SNN(n_inputs, n_hidden, n_outputs, beta=0.9).to(device)
loss_hist, target_list, clapp_loss_hist = train_sample_wise(SNN, train_loader, epochs, device)

torch.save(SNN.state_dict(), f'models/{model_name}.pt')

# test model
losses, loss_per_digit, clapp_activation, target_list, clapp_losses = test_old(SNN, test_loader, device)
print(loss_per_digit)
print('Mean Loss:', losses.sum()/10000)



