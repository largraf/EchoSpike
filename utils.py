import torch

def train(net, trainloader, epochs, device, model_name, batch_size=1, freeze=[], online=False, lr=1e-5):
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
    accuracies = []
    print_interval = 100*batch_size if 'mnist' in model_name else 40*batch_size
    current_epoch_loss = 1e5 # some large number
    # training loop
    optimizer = torch.optim.SGD([{"params":par.fc.parameters(), 'lr': lr} for par in net.layers])
    optimizer.zero_grad()
    net.train()
    bf = 0
    target = [torch.randint(trainloader.num_classes, (1,)).item() for _ in range(batch_size)]
    spks = torch.zeros(len(net.layers)+1, device=device)

    while True:
        # Train loop
        data, target = trainloader.next_item(target, contrastive=(bf==-1))
        data = data.float().to(device)
        target = target.to(device)
        sample_loss = torch.zeros(len(net.layers), device=device)

        for step in range(data.shape[0]):
            # iterate over time steps
            if online:
                inp_activity = data[step].mean(axis=-1)
            else:
                inp_activity = None
            spk, _, loss = net(data[step], torch.tensor(bf, device=device), freeze, inp_activity=inp_activity)
            spks += torch.stack([data[step].mean(), *[sp.mean() for sp in spk]])    # to analyze nr of spks
            sample_loss += loss
            if online:
                optimizer.step()
                optimizer.zero_grad()

        loss_hist.append(sample_loss/data.shape[0]) 
        accuracies.append(net.reset(bf))

        if bf == -1 and not online:
            # update weigths after one predictive and one contrastive batch, before weight update
            optimizer.step()
            optimizer.zero_grad()
        bf = 1 if bf != 1 else -1

        step = len(loss_hist) * batch_size
        epoch = step // len(trainloader)
        if step % print_interval < batch_size and len(loss_hist) > 1:
            # print loss and accuracy
            print(f"Epoch {epoch}, Step {step} \nEchoSpike Loss: {torch.stack(loss_hist[-print_interval//batch_size:]).mean(axis=0)}")
            print(f"Acc: {torch.stack(accuracies[-print_interval//batch_size:]).mean(axis=0)}")
            print(f"Spks: {spks*batch_size/print_interval}")
            spks = torch.zeros(len(net.layers)+1, device=device)
        if epoch >= epochs:
            break
        if step % len(trainloader) < batch_size and epoch % 20 == 0:
            # save checkpoint if performance improves
            last_epoch_loss = current_epoch_loss
            current_epoch_loss = torch.stack(loss_hist[-len(trainloader)//batch_size:]).mean().item()
            print(f'epoch loss: {current_epoch_loss}')
            if current_epoch_loss < last_epoch_loss:
                torch.save(net.state_dict(), f'models/{model_name}_epoch{epoch}.pt')
            
    return torch.stack(loss_hist)


def test(net, testloader, device, batch_size=1):
    torch.set_grad_enabled(False)
    net.eval()
    spk_history = []
    target_list = []
    losses = []

    bf = 0
    target = [torch.randint(testloader.num_classes, (1,)).item() for _ in range(batch_size)]
    while True:
        data, target = testloader.next_item(target, contrastive=(bf==-1))
        target_list.append(target)
        data = data.float().to(device)
        target = target.to(device)
        logit_list = []
        activation_list = []
        loss_sample = torch.zeros(len(net.layers), device=device)
        for step in range(data.shape[0]):
            out_spk, _, loss = net(data[step], torch.tensor(bf, device=device))
            logit_list.append(out_spk[-1])
            activation_list.append(out_spk)
            loss_sample += loss

        losses.append(loss_sample)
        spk_history.append(activation_list[0])
        for i in range(1, len(activation_list)):
            for l in range(len(spk_history[-1])):
                spk_history[-1][l] += activation_list[i][l]
        net.reset(bf)
        bf = 1 if bf != 1 else -1
        if len(losses)*batch_size > len(testloader):
            break
    return spk_history, target_list, losses

def get_accuracy(SNN, out_projs, dataloader, device):
    """Get the accuracy of the SNN on the given dataset.

    Args:
        SNN (EchoSpike): The SNN model without the output projection.
        out_projs (list): output projections directly from the inputs and from each layer.
        dataloader (classwise_loader): The classwise dataloader for the dataset.
        device (torch.device): The device to use for training the model.

    Returns:
        list: a list of accuracies for each output projection.
        torch.Tensor: the prediction matrix.
    """
    from tqdm.notebook import trange
    batch_size = dataloader.batch_size
    correct = torch.zeros(len(out_projs))
    for out_proj in out_projs:
        out_proj.eval()
    no_spk = True
    total = 0
    SNN.eval()
    pred_matrix = torch.zeros(dataloader.num_classes, dataloader.num_classes)
    for idx in trange(0, len(dataloader), batch_size):
        for out_proj in out_projs:
            out_proj.reset()
        SNN.reset(0)
        inp, target = dataloader.x[idx:idx+batch_size], dataloader.y[idx:idx+batch_size]
        logits = len(out_projs)*[torch.zeros((inp.shape[0],20))]
        for step in range(inp.shape[1]):
            data_step = inp[:,step].float().to(device)
            spk_step, _, _ = SNN(data_step, 0)
            spk_step = [data_step, *spk_step]
            for i, out_proj in enumerate(out_projs):
                out, mem = out_proj(spk_step[i], target)
                if no_spk:
                    logits[i] = mem
                else:
                    logits[i] = logits[i] + out
        for i, logit in enumerate(logits):
            pred = logit.argmax(axis=-1)
            correct[i] += int((pred == target).sum())
            total += len(pred)
        # for the last layer create the prediction matrix
        for j in range(pred.shape[0]):
            pred_matrix[int(target[j]), int(pred[j])] += 1

    assert total == len(dataloader)
    correct /= len(dataloader)
    print('Directly from inputs:')
    print(f'Accuracy: {100*correct[0]:.2f}%')
    accs = [correct[0]]
    for i in range(len(out_projs)-1):
        print(f'From layer {i+1}:')
        print(f'Accuracy: {100*correct[i+1]:.2f}%')
        accs.append(correct[i+1])

    return accs, pred_matrix