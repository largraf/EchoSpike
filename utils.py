import tonic
from tonic import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import snntorch.functional as SF

def load_NMNIST(n_time_bins, batch_size=1):
    # load NMNIST dataset
    sensor_size = tonic.datasets.NMNIST.sensor_size
    print(sensor_size)
    transf = [transforms.Denoise(filter_time=10000),
              transforms.ToFrame(sensor_size=sensor_size,
                                 n_time_bins=n_time_bins)]
    frame_transform = transforms.Compose(transf)

    trainset = tonic.datasets.NMNIST(save_to='./data',
                                     transform=frame_transform, train=True)
    testset = tonic.datasets.NMNIST(save_to='./data',
                                    transform=frame_transform, train=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def train(net, trainloader, epochs, device):
    acc_hist = []
    loss_hist = []
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    # training loop
    prev_target = -1
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    net.train()
    mem_history = []
    target_list = []
    for epoch in range(epochs):
        for i, (data, targets) in tqdm(enumerate(iter(trainloader))):
            optimizer.zero_grad()
            data = data.squeeze(0).float().to(device)
            target_list.append(targets)
            targets = targets.to(device)

            bf = 1 if targets == prev_target else -1
            prev_target = targets
            logit_list, mem_his = net(data, targets, bf)
            mem_history.append(mem_his)
            loss_val = loss_fn(torch.tensor(logit_list), targets)
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item()) 

            if i % 500 == 0:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
                if i == 2000:
                    break
    return loss_hist, torch.cat(mem_history, 0), target_list

def test(net, testloader, device):
    correct = 10*[0]
    net.eval()
    mem_history = []
    target_list = []
    with torch.no_grad():
        for i, (data, targets) in tqdm(enumerate(iter(testloader))):
            data = data.squeeze(0).float().to(device)
            targets = targets.to(device)
            logit_list, mem_his = net(data, targets, 1)
            mem_history.append(mem_his)
            target_list.append(targets)
            pred = torch.tensor(logit_list).sum(axis=0).argmax(axis=-1)
            bool_idx = pred == targets
            for val in targets[bool_idx]:
                correct[val] += 1
    return correct, mem_history, target_list
