import tonic
from tonic import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from tqdm import tqdm

# load NMNIST dataset
sensor_size = tonic.datasets.NMNIST.sensor_size
n_time_bins = 32
print(sensor_size)
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         n_time_bins=n_time_bins)])

batch_size = 16
trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

for data in []:#train_loader:
    print(data[0].shape, data[1])
    for i in range(n_time_bins):
        plt.imshow(data[0][0][i][0] + data[0][0][i][1])
        plt.pause(1)
    break

beta = 0.99

num_inputs = 2312
num_hidden = 512
num_outputs = 10

device = 'cpu'

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.rlif1 = snn.RLeaky(beta=beta, linear_features=num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        spk1, mem1 = self.rlif1.init_rleaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []  # Record the output trace of spikes
        mem2_rec = []  # Record the output trace of membrane potential

        for step in range(n_time_bins):
            x_flat = x[:, step].view(x.shape[0], -1)
            cur1 = self.fc1(x_flat)
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = Net().to(device)

n_epochs = 1

def training(net, trainloader, epochs):
    acc_hist = []
    loss_hist = []
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    # training loop
    for epoch in range(n_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.float().to(device)
            targets = targets.to(device)

            net.train()
            spk_rec, _ = net(data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")
    return acc_hist, loss_hist

def testing(net, testloader):
    correct = np.zeros(10)
    acc = 0
    net.eval()
    with torch.no_grad():
        for i, (data, targets) in tqdm(enumerate(iter(testloader))):
            data = data.float().to(device)
            targets = targets.to(device)
            spk_rec, _ = net(data)
            pred = spk_rec.sum(axis=0).argmax(axis=-1)
            bool_idx = pred == targets
            for val in targets[bool_idx]:
                correct[val] += 1
    return correct



acc_hist, loss_hist = training(net, train_loader, n_epochs)
plt.plot(acc_hist)
plt.plot(loss_hist)
plt.show()
correct = testing(net, test_loader)
print(correct)
print('Accuracy:', sum(correct)/10000)
