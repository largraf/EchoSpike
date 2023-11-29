from utils import train_shd_segmented, train_samplewise_clapp
from data import load_SHD
from model import CLAPP_SRNN
import torch

device = 'cpu'
epochs = 3
batch_size = 20
n_inputs = 700 # 28*28
n_hidden = 3 * [512]
n_outputs = 20
model_name = 'SNN_CLAPP_test'

# load dataset
n_time_bins = 100
train_loader, test_loader = load_SHD(batch_size=batch_size)#, scale=0.4) #load_NMNIST(n_time_bins, batch_size=batch_size)

# train and save model
SNN = CLAPP_SRNN(n_inputs, n_hidden, n_outputs, beta=0.95, out_proj=False, device='cpu', recurrent=True).to(device)
# SNN.load_state_dict(torch.load(f'models/SHD_3x512_batch20_mse.pt', map_location=device))
clapp_loss_hist = train_samplewise_clapp(SNN, train_loader, epochs, device, model_name, batch_size=batch_size, temporal=True)

torch.save(SNN.state_dict(), f'models/{model_name}.pt')
torch.save(clapp_loss_hist, f'models/{model_name}_clapp_loss.pt')

# test model
# losses, loss_per_digit, clapp_activation, target_list, clapp_losses = test_old(SNN, test_loader, device)
# print(loss_per_digit)
# print('Mean Loss:', losses.sum()/10000)



