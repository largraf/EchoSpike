import torch

def train(net, trainloader, epochs, device, model_name, batch_size=1, freeze=[], temporal=True, online=False):
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
    torch.manual_seed(123)
    clapp_loss_hist = []
    clapp_accuracies = []
    print_interval = 10*batch_size
    current_epoch_loss = 1e5 # some large number
    # training loop
    online_factor = 1e-2 if online else 1
    optimizer_clapp = torch.optim.SGD([{"params":par.fc.parameters(), 'lr': online_factor*1e-3} for par in net.clapp])
    optimizer_clapp.zero_grad()
    net.train()
    bf = 0
    target = [torch.randint(trainloader.num_classes, (1,)).item() for _ in range(batch_size)]
    spks = torch.zeros(len(net.clapp)+1, device=device)
    while True:
        data, target = trainloader.next_item(target, contrastive=(bf==-1))
        data = data.float().to(device)
        target = target.to(device)
        if temporal:
            clapp_sample_loss = torch.zeros(len(net.clapp), device=device)

        for step in range(data.shape[0]):
            factor = bf if step == data.shape[0]-1 else 0
            if temporal:
                factor = bf
            spk, _, clapp_loss = net(data[step], None, torch.tensor(factor, device=device), freeze)
            spks += torch.stack([data[step].mean(), *[sp.mean() for sp in spk]])    # to analyze nr of spks
            if temporal:
                clapp_sample_loss += clapp_loss
            if online:
                optimizer_clapp.step()
                optimizer_clapp.zero_grad()
        if temporal:
            clapp_loss_hist.append(clapp_sample_loss/data.shape[0]) 
        else:
            clapp_loss_hist.append(clapp_loss)
        clapp_accuracies.append(net.reset(bf))
        if bf == -1 and not online:
            # ensure that there was one predictive and one contrastive batch, before weight update
            optimizer_clapp.step()
            optimizer_clapp.zero_grad()
        bf = 1 if bf != 1 else -1
        step = len(clapp_loss_hist) * batch_size
        epoch = step // len(trainloader)
        if step % print_interval < batch_size and len(clapp_loss_hist) > 1:
            print(f"Epoch {epoch}, Step {step} \nCLAPP Loss: {torch.stack(clapp_loss_hist[-print_interval//batch_size:]).mean(axis=0)}")
            print(f"CLAPP Acc: {torch.stack(clapp_accuracies[-print_interval//batch_size:]).mean(axis=0)}")
            print(f"Spks: {spks*batch_size/print_interval}")
            spks = torch.zeros(len(net.clapp)+1, device=device)
        if epoch >= epochs:
            break
        if step % len(trainloader) < batch_size and (epoch + 1) % 5 == 0:
            # save checkpoint if performance improves
            last_epoch_loss = current_epoch_loss
            current_epoch_loss = torch.stack(clapp_loss_hist[-len(trainloader)//batch_size:]).mean().item()
            print(f'epoch loss: {current_epoch_loss}')
            if current_epoch_loss < last_epoch_loss:
                torch.save(net.state_dict(), f'models/{model_name}_epoch{epoch}.pt')
            
    return torch.stack(clapp_loss_hist)


def test(net, testloader, device, batch_size=1, temporal=False):
    torch.set_grad_enabled(False)
    net.eval()
    spk_history = []
    target_list = []
    clapp_losses = []

    bf = 0
    target = [torch.randint(testloader.num_classes, (1,)).item() for _ in range(batch_size)]
    while True:
        data, target = testloader.next_item(target, contrastive=(bf==-1))
        target_list.append(target)
        data = data.float().to(device)
        target = target.to(device)
        logit_list = []
        activation_list = []
        if temporal:
            clapp_loss_sample = torch.zeros(len(net.clapp), device=device)
        for step in range(data.shape[0]):
            factor = bf if step == data.shape[0]-1 else 0
            if temporal:
                factor = bf
            out_spk, activations, clapp_loss = net(data[step], target, torch.tensor(factor, device=device))
            logit_list.append(out_spk[-1])
            activation_list.append(torch.stack(out_spk))
            if factor != 0 and not temporal:
                clapp_losses.append(clapp_loss)
            elif temporal:
                clapp_loss_sample += clapp_loss
        if temporal:
            clapp_losses.append(clapp_loss_sample)
        spk_history.append(torch.stack(activation_list).sum(axis=0))
        net.reset(bf)
        coin_flip = torch.rand(1) > 0.5
        if coin_flip:
            bf = -1
        else:
            bf = 1
        if len(clapp_losses)*batch_size > len(testloader):
            break
    return spk_history, target_list, clapp_losses
