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
    for epoch in range(epochs):
        for i, (data, targets) in tqdm(enumerate(iter(trainloader))):
            data = data.squeeze(0).float().to(device)
            targets = targets.to(device)

            net.train()
            bf = 1 if targets == prev_target else -1
            prev_target = targets
            spk_rec, _ = net(data, bf)
            loss_val = loss_fn(spk_rec, targets)
            if net.rule == 'backprop':
                # Gradient calculation + weight update
                optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item()) 

            acc = SF.accuracy_rate(spk_rec.unsqueeze(0), targets)
            acc_hist.append(acc)
            if i % 500 == 0:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
                print(f"Accuracy: {acc * 100:.2f}%\n")
    return acc_hist, loss_hist

def test(net, testloader, device):
    correct = 10*[0]
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
