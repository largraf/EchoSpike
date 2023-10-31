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

def train_shd_segmented(net, trainloader, epochs, device, segment_size=20, batch_size=1, freeze=0):
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
    clapp_loss_hist = []
    # training loop
    optimizer_clapp = torch.optim.Adamax([{"params":par.fc.parameters(), 'lr': 2e-4} for par in net.clapp] +
                                       [{"params": par.pred.parameters(), 'lr': 2e-4} for par in net.clapp])
    optimizer_clapp.zero_grad()
    net.train()
    target = torch.randint(trainloader.num_classes, (1,)).item()
    while True:
        # Choose the next random target that is different from current target
        data, target = trainloader.next_item(int(target), contrastive=True)
        net.reset()
        data = data.squeeze(0).float().to(device)
        target = target.to(device)
        predict_segment = torch.randint(data.shape[0]//segment_size - 1, (1,)).item() + 1

        for step in range(data.shape[0]):
            if step == segment_size * (predict_segment-1) - 1:
                event = 'predict'
            elif step == segment_size * predict_segment - 1:
                event = 'evaluate'
            else:
                event = None
            _, _, clapp_loss = net(data[step].flatten(), target, event)
            if event == 'evaluate':
                clapp_loss_hist.append(clapp_loss)
                if len(clapp_loss_hist) % batch_size == 0:
                    optimizer_clapp.step()
                    optimizer_clapp.zero_grad()
                break

        epoch = len(clapp_loss_hist) // len(trainloader)
        if len(clapp_loss_hist) % 1000 == 0 and len(clapp_loss_hist) > 1:
            print(f"Epoch {epoch}, Iteration {len(clapp_loss_hist)} \nCLAPP Loss: {torch.stack(clapp_loss_hist[-1000:]).sum(axis=0)/1000}")
        if epoch >= epochs:
            break
    return torch.stack(clapp_loss_hist)

def train_samplewise_clapp(net, trainloader, epochs, device, model_name, batch_size=1, freeze=0):
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
    current_epoch_loss = 1e5 # some large number
    # training loop
    optimizer_clapp = torch.optim.SGD([{"params":par.fc.parameters(), 'lr': 1e-3} for par in net.clapp] +
                                       [{"params": par.pred.parameters(), 'lr': 1e-3} for par in net.clapp])
    optimizer_clapp.zero_grad()
    net.train()
    bf = 0
    target = torch.randint(trainloader.num_classes, (1,)).item()
    while True:
        data, target = trainloader.next_item(int(target), contrastive=(bf==-1))
        net.reset()
        data = data.squeeze(0).float().to(device)
        logits_per_step = []
        target = target.to(device)

        for step in range(data.shape[0]):
            factor = bf if step == data.shape[0]-1 else 0
            _, _, clapp_loss = net(data[step].flatten(), target, torch.tensor(factor, device=device))
            if factor != 0:
                clapp_loss_hist.append(clapp_loss)
                if len(clapp_loss_hist) % batch_size == 0:
                    optimizer_clapp.step()
                    optimizer_clapp.zero_grad()
        coin_flip = torch.rand(1) > 0.5
        if coin_flip:
            bf = -1
        else:
            bf = 1

        epoch = len(clapp_loss_hist) // len(trainloader)
        if len(clapp_loss_hist) % 5000 == 0 and len(clapp_loss_hist) > 1:
            print(f"Epoch {epoch}, Iteration {len(clapp_loss_hist)} \nCLAPP Loss: {torch.stack(clapp_loss_hist[-5000:]).sum(axis=0)/5000}")
        if epoch >= epochs:
            break
        if len(clapp_loss_hist) % len(trainloader) == 0 and(epoch + 1) % 5 == 0:
            # save checkpoint if performance improves
            last_epoch_loss = current_epoch_loss
            current_epoch_loss = torch.stack(clapp_loss_hist[-len(trainloader):]).mean().item()
            print(f'epoch loss: {current_epoch_loss}')
            if current_epoch_loss < last_epoch_loss:
                torch.save(net.state_dict(), f'models/{model_name}_epoch{epoch}.pt')
            
    return torch.stack(clapp_loss_hist)


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
            logit_list.append(out_spk[-1])
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
    if net.has_out_proj:
        print(f'Accuracy: {100*sum(true_false)/len(true_false):.2f}%')
    return torch.stack(loss_list), loss_per_class, mem_history, target_list, clapp_losses

def train_out_projection(SNN, train_loader, epochs, device, from_layer=-1):
    # Gradient calculation + weight update
    torch.set_grad_enabled(False)
    import snntorch.functional as SF
    losses_out = []
    SNN.out_proj.out_proj.reset_parameters()
    SNN.clapp = torch.nn.ModuleList(SNN.clapp[:-2])
    print(SNN.out_proj.out_proj.weight.abs().mean())
    optimizer = torch.optim.Adam(SNN.out_proj.parameters(), lr=1e-4)
    SNN.train()
    mem_his_list = []
    target = 0
    correct = 0
    acc = []
    with torch.no_grad():
        while True:
            data, target = train_loader.next_item(target, contrastive=True)
            SNN.reset()
            logit_list = []
            data = data.squeeze()
            optimizer.zero_grad()
            for step in range(data.shape[0]):
                data_step = data[step].float().to(device)
                target = target.to(device)
                logits, mem_his, clapp_loss = SNN(data_step, target, 0)
                mem_his_list.append(mem_his)
                logit_list.append(logits)
            optimizer.step()
            
            pred = torch.stack(logit_list).sum(axis=0)
            if pred.max() < 1: print(pred.max())
            correct = correct + 1 if pred.argmax() == target else correct 
            losses_out.append(torch.nn.functional.cross_entropy(pred, target.squeeze().long()))
            if len(losses_out) % 100 == 0:
                print(len(losses_out), sum(losses_out[-100:])/100)
                print(SNN.out_proj.out_proj.weight.abs().mean())
                print(pred.max())
                print(f'Correct: {correct}%')
                acc.append(correct/100)
                correct = 0
            if len(losses_out) > 5000:
                break
    print(sum(acc)/len(acc))
    return losses_out, acc
