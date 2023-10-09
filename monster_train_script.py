# %% [markdown]
# # SNN-CLAPP

# %%
import matplotlib.pyplot as plt
from utils import load_NMNIST, load_PMNIST, train, test
from model import CLAPP_SNN
import numpy as np
import torch

device = 'cuda'
epochs = 2
batch_size = 1
n_inputs = 28*28 #34 * 34 * 2
n_hidden = 2 * [512]
n_outputs = 10
model_name = 'SNN_CLAPP_0'


# %% [markdown]
# ## Dataset
# N-MNIST 
# 

# %%
n_time_bins = 4
train_loader, test_loader = load_PMNIST(n_time_bins, scale=0.9) #load_NMNIST(n_time_bins, batch_size=batch_size)

# Plot Example
frames, target = next(iter(train_loader))
print(frames.shape, target)
for i in range(n_time_bins):
    plt.figure()
    plt.imshow(frames[0,i].view(28,28), cmap='gray')
    plt.colorbar()


# %% [markdown]
# ## Training

# %%

SNN = CLAPP_SNN(n_inputs, n_hidden, n_outputs).to(device)
loss_hist, target_list, clapp_loss_hist = train(SNN, train_loader, epochs, device)

# %%
losses, loss_per_digit, clapp_activation, target_list = test(SNN, test_loader, device)
print(loss_per_digit)
print('Mean Loss:', losses.sum()/10000)

torch.save(SNN.state_dict(), f'models/{model_name}.pt')


