import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import snntorch.functional as SF
import snntorch as snn

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

def train_shd_supervised_clapp(net, trainloader, epochs, device):
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
    optimizer_clapp = torch.optim.Adam([{"params":par.fc.parameters(), 'lr': 1e-3} for par in net.clapp] +
                                       [{"params": par.pred.parameters(), 'lr': 1e-4} for par in net.clapp])
    optimizer_out = torch.optim.Adam(net.out_proj.parameters(), lr=1e-4)
    net.train()
    target_list = []
    bf = 0
    target = torch.randint(trainloader.num_classes, (1,)).item()
    while True:
        data, target = trainloader.next_item(int(target), contrastive=(bf==-1))
        net.reset()
        data = data.squeeze(0).float().to(device)
        logits_per_step = []
        target_list.append(target)
        target = target.to(device)

        for step in range(data.shape[0]):
            factor = bf if step == data.shape[0]-1 else 0
            if factor != 0:
                optimizer_clapp.zero_grad()
                optimizer_out.zero_grad()
            out_spk, _, clapp_loss = net(data[step].flatten(), target, torch.tensor(factor, device=device))
            logits_per_step.append(out_spk)
            if factor != 0:
                clapp_loss_hist.append(clapp_loss)
                optimizer_clapp.step()
                optimizer_out.step()
        coin_flip = torch.rand(1) > 0.5
        if coin_flip:
            bf = -1
        else:
            bf = 1

        # loss_val = loss_fn(torch.stack(logits_per_step).unsqueeze(1), targets)
        # Store loss history for future plotting
        # loss_hist.append(loss_val.item()) 

        epoch = len(clapp_loss_hist) // len(trainloader)
        if len(clapp_loss_hist) % 200 == 0 and len(clapp_loss_hist) > 1:
            print(f"Epoch {epoch}, Iteration {len(clapp_loss_hist)} \nTrain Loss: {sum([0])/200:.2f} \nCLAPP Loss: {torch.stack(clapp_loss_hist[-200:]).sum(axis=0)/200}")
        if epoch >= epochs:
            break
    return loss_hist, target_list, torch.stack(clapp_loss_hist)

def train_sample_wise(net, trainloader, epochs, device):
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
    optimizer_clapp = torch.optim.SGD([{"params":par.fc.parameters(), 'lr': 1e-2} for par in net.clapp] +
                                       [{"params": par.pred.parameters(), 'lr': 1e-3} for par in net.clapp])
    optimizer_out = torch.optim.SGD(net.out_proj.parameters(), lr=1e-4)
    net.train()
    target_list = []
    bf = 0
    for epoch in range(epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            if targets == prev_target:
                continue
            while True:
                net.reset()
                data = data.squeeze(0).float().to(device)
                logits_per_step = []

                for step in range(data.shape[0]):
                    target_list.append(targets)
                    targets = targets.to(device)
                    factor = bf if step == data.shape[0]-1 else 0
                    if factor != 0:
                        optimizer_clapp.zero_grad()
                        optimizer_out.zero_grad()
                    out_spk, _, clapp_loss = net(data[step].flatten(), targets, torch.tensor(factor, device=device))
                    logits_per_step.append(out_spk)
                    if factor != 0:
                        clapp_loss_hist.append(clapp_loss)
                        optimizer_clapp.step()
                        optimizer_out.step()
                coin_flip = torch.rand(1) > 0.5
                if coin_flip:
                    bf = -1
                    break
                else:
                    bf = 1
    
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

def test_sample_wise(net, testloader, device):
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
            net.reset()
            data = data.squeeze(0).float().to(device)
            targets = targets.to(device)
            logit_list = []
            for step in range(data.shape[0]):
                logits, mem_his, clapp_loss = net(data[step].flatten(), targets, bf)
                if bf != 0:
                    clapp_losses.append(clapp_loss)
                    mem_history.append(mem_his)
                    target_list.append(targets)
                logit_list.append(logits)
                # clapp_loss_hist.append(clapp_loss)
                if step == data.shape[0] - 2:
                    bf = -1
                else: bf = 0
            pred = torch.stack(logit_list).sum(axis=0)
            loss = torch.nn.functional.cross_entropy(pred, targets.squeeze())
            loss_per_class[targets] += loss
            loss_list.append(loss)
    return torch.stack(loss_list), loss_per_class, mem_history, target_list, clapp_losses

def test_SHD(net, testloader, device):
    torch.set_grad_enabled(False)
    loss_per_class = testloader.num_classes*[0]
    net.eval()
    mem_history = []
    target_list = []
    loss_list = []
    clapp_losses = []
    true_false = []

    bf = 0
    target = torch.randint(testloader.num_classes, (1,)).item()
    while True:
        data, target = testloader.next_item(int(target), contrastive=(bf==-1))
        target_list.append(target)
        net.reset()
        data = data.squeeze(0).float().to(device)
        target = target.to(device)
        logit_list = []
        activation_list = []
        for step in range(data.shape[0]):
            factor = bf if step == data.shape[0]-1 else 0
            out_spk, activations, clapp_loss = net(data[step].flatten(), target, torch.tensor(factor, device=device))
            logit_list.append(out_spk)
            activation_list.append(activations)
            if factor != 0:
                clapp_losses.append(clapp_loss)
        mem_history.append(torch.stack(activation_list).sum(axis=0))
        coin_flip = torch.rand(1) > 0.5
        if coin_flip:
            bf = -1
        else:
            bf = 1
        pred = torch.stack(logit_list).sum(axis=0)
        prediction = torch.argmax(pred)
        if prediction.sum() > 0:
            pass
        true_false.append(prediction == target.squeeze())
        loss = torch.nn.functional.cross_entropy(pred, target.squeeze().long())
        loss_per_class[target.long()] += loss
        loss_list.append(loss)
        if len(loss_list) > len(testloader):
            break
    return torch.stack(loss_list), loss_per_class, mem_history, target_list, clapp_losses