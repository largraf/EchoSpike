import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import snntorch.functional as SF
import snntorch as snn

def load_NMNIST(n_time_bins, batch_size=1):
    """
    Load the Neuromorphic-MNIST dataset.

    Parameters:
        n_time_bins (int): The number of time bins per digit.
        batch_size (int, optional): The batch size. Defaults to 1.

    Returns:
        train_loader (DataLoader): The data loader for the training set.
        test_loader (DataLoader): The data loader for the test set.
    """
    import tonic
    from tonic import transforms
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

def load_PMNIST(n_time_steps, batch_size=1, scale=1, patches=False):
    """
    Load the Poisson spike encoded MNIST dataset with the specified parameters.

    Parameters:
        n_time_steps (int): The number of time steps for the spike encoding.
        batch_size (int, optional): The batch size for the data loaders. Default is 1.
        scale (int, optional): The scaling factor for the spike encoding. Default is 1.
        patches (bool, optional): Whether to load the dataset in 4 patches or not. Default is False.

    Returns:
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
    """
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: snn.spikegen.rate(x*scale, n_time_steps).view(n_time_steps, -1)])
    if patches:
        transform = transforms.Compose([transforms.ToTensor(),
                                        lambda x: torch.stack([x[:,:14,:14], x[:,:14,14:], x[:,14:,14:], x[:,14:,:14]])])
    trainset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                       transform=transform,  
                                           download=True)
    testset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                       transform=transform,  
                                           download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, 
                                            shuffle=False) 
    return train_loader, test_loader

def load_half_MNIST(batch_size=1):
    """
    Load the MNIST dataset and split images into two halfs.
    
    Parameters:
        batch_size (int): The number of samples per batch. Default is 1.
        
    Returns:
        train_loader (DataLoader): A DataLoader object for the train set.
        test_loader (DataLoader): A DataLoader object for the test set.
    """
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: torch.stack([x[:,:,:14], x[:,:,14:]])])
    trainset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                       transform=transform,  
                                           download=True)
    testset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                       transform=transform,  
                                           download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, 
                                            shuffle=False) 
    return train_loader, test_loader

def train(net, trainloader, epochs, device):
    """
    Trains a SNN.

    Args:
        net (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        epochs (int): The number of epochs for training.
        device (torch.device): The device to use for training the model.

    Returns:
        tuple: A tuple containing the following:
            - loss_hist (list): A list of the loss values during training.
            - mem_history (torch.Tensor): A tensor containing the LIF memory history.
            - target_list (list): A list of the target values.
    """
    torch.set_grad_enabled(False)
    loss_hist = []
    clapp_loss_hist = []
    loss_fn = SF.ce_count_loss()
    # training loop
    prev_target = -1
    optimizer_clapp = torch.optim.SGD([{"params":par.fc.parameters(), 'lr': 1e-3} for par in net.clapp] +
                                       [{"params": par.pred.parameters(), 'lr': 1e-5} for par in net.clapp])
    optimizer_out = torch.optim.SGD(net.out_proj.parameters(), lr=1e-5)
    net.train()
    target_list = []
    bf = 0
    for epoch in range(epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            # net.reset()
            data = data.squeeze(0).float().to(device)
            if targets == prev_target:
                continue
            logits_per_step = []
            for step in range(data.shape[0]):
                # net.reset()
                optimizer_clapp.zero_grad()
                optimizer_out.zero_grad()
                target_list.append(targets)
                targets = targets.to(device)

                logit_list, _, clapp_loss = net(data[step].flatten(), targets, torch.tensor(bf, device=device))
                logits_per_step.append(logit_list)
                if bf != 0:
                    clapp_loss_hist.append(clapp_loss)
                    optimizer_clapp.step()
                optimizer_out.step()
                coin_flip = torch.rand(1) > 0.5
                if coin_flip or step == data.shape[0]-1:
                    bf = -1
                    break
                else:
                    bf = 1

            if i % 1000 == 0 and i > 1:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {sum([0])/1000:.2f} \nCLAPP Loss: {torch.stack(clapp_loss_hist[-2000:]).sum(axis=0)/2000}")
            prev_target = targets.cpu()
    return loss_hist, target_list, torch.stack(clapp_loss_hist)

def train_sequences(net, trainloader, epochs, device):
    """
    Trains a SNN.

    Args:
        net (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        epochs (int): The number of epochs for training.
        device (torch.device): The device to use for training the model.

    Returns:
        tuple: A tuple containing the following:
            - loss_hist (list): A list of the loss values during training.
            - mem_history (torch.Tensor): A tensor containing the LIF memory history.
            - target_list (list): A list of the target values.
    """
    torch.set_grad_enabled(False)
    loss_hist = []
    clapp_loss_hist = []
    loss_fn = SF.ce_count_loss()
    # training loop
    prev_target = -1
    optimizer_clapp = torch.optim.SGD([{"params":par.fc.parameters(), 'lr': 1e-3} for par in net.clapp] +
                                       [{"params": par.pred.parameters(), 'lr': 1e-5} for par in net.clapp])
    optimizer_out = torch.optim.SGD(net.out_proj.parameters(), lr=1e-5)
    net.train()
    target_list = []
    for epoch in range(epochs):
        bf = 0
        for i, (data, targets) in enumerate(iter(trainloader)):
            net.reset()
            data = data.squeeze(0).float().to(device)
            if targets == prev_target:
                continue
            logits_per_step = []

            for step in range(data.shape[0]):
                optimizer_clapp.zero_grad()
                optimizer_out.zero_grad()
                target_list.append(targets)
                targets = targets.to(device)

                out_spk, _, clapp_loss = net(data[step].flatten(), targets, torch.tensor(bf, device=device))
                logits_per_step.append(out_spk)
                if bf != 0:
                    clapp_loss_hist.append(clapp_loss)
                    optimizer_clapp.step()
                optimizer_out.step()
                if step == data.shape[0] - 1:
                    bf = -1
                elif step == data.shape[0] - 2:
                    bf = 1
                else:
                    bf = 0
            # loss_val = loss_fn(torch.stack(logits_per_step).unsqueeze(1), targets)
            # Store loss history for future plotting
            # loss_hist.append(loss_val.item()) 

            if i % 1000 == 0 and i > 1:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {sum([0])/1000:.2f} \nCLAPP Loss: {torch.stack(clapp_loss_hist[-2000:]).sum(axis=0)/2000}")
            prev_target = targets.cpu()
    return loss_hist, target_list, torch.stack(clapp_loss_hist)

def test(net, testloader, device):
    torch.set_grad_enabled(False)
    loss_per_class = 10*[0]
    net.eval()
    mem_history = []
    target_list = []
    loss_list = []
    clapp_losses = []

    with torch.no_grad():
        bf = 0
        for i, (data, targets) in enumerate(iter(testloader)):
            net.reset()
            data = data.squeeze(0).float().to(device)
            targets = targets.to(device)
            logit_list = []
            for step in range(data.shape[0]):
                logits, mem_his, clapp_loss = net(data[step].flatten(), targets, bf)
                if step == 1:
                    clapp_losses.append(clapp_loss)
                    mem_history.append(mem_his)
                    target_list.append(targets)
                logit_list.append(logits)
                # clapp_loss_hist.append(clapp_loss)
            pred = torch.stack(logit_list).sum(axis=0)
            loss = torch.nn.functional.cross_entropy(pred, targets.squeeze())
            loss_per_class[targets] += loss
            loss_list.append(loss)
    return torch.stack(loss_list), loss_per_class, mem_history, target_list, clapp_losses

def test_old(net, testloader, device):
    torch.set_grad_enabled(False)
    loss_per_class = 10*[0]
    net.eval()
    mem_history = []
    target_list = []
    loss_list = []
    clapp_losses = []

    bf = 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(iter(testloader)):
            data = data.squeeze(0).float().to(device)
            targets = targets.to(device)
            logit_list = []
            for step in range(data.shape[0]):
                net.reset()
                logits, mem_his, clapp_loss = net(data[step].flatten(), targets, bf)
                if bf != 0:
                    clapp_losses.append(clapp_loss)
                logit_list.append(logits)
                mem_history.append(mem_his)
                # clapp_loss_hist.append(clapp_loss)
                target_list.append(targets)
                if step == data.shape[0] - 1:
                    bf = -1
                elif step == data.shape[0] - 2:
                    bf = 1
                else: bf = 0
            pred = torch.stack(logit_list).sum(axis=0)
            loss = torch.nn.functional.cross_entropy(pred, targets.squeeze())
            loss_per_class[targets] += loss
            loss_list.append(loss)
    return torch.stack(loss_list), loss_per_class, mem_history, target_list, clapp_losses
