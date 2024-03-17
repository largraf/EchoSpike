import torch
from data import augment_shd
import numpy as np
from tqdm.notebook import trange
from model import simple_out

def train(net, trainloader, epochs, device, model_name, batch_size=1, freeze=[], online=False, lr=1e-5, augment=False):
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
        if augment:
            data = augment_shd(data)
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
            print(f"Acc: {torch.stack(accuracies).mean(axis=0)}")
            accuracies = []
            print(f"Spks: {spks*batch_size/print_interval}")
            spks = torch.zeros(len(net.layers)+1, device=device)
        if epoch >= epochs:
            break
        if step % len(trainloader) < batch_size and epoch % 20 == 0:
            # save checkpoint
            current_epoch_loss = torch.stack(loss_hist[-20*len(trainloader)//batch_size:]).mean().item()
            print(f'epoch loss: {current_epoch_loss}')
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

def get_accuracy(SNN, out_projs, dataloader, device, cat=False):
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
                if cat and i > 0:
                    out, mem = out_proj(torch.cat(spk_step[:i+1], axis=-1), target)
                else:
                    out, mem = out_proj(spk_step[i], target)
                if step == inp.shape[1]-1:
                    logits[i] = mem
        for i, logit in enumerate(logits):
            pred = logit.argmax(axis=-1)
            correct[i] += int((pred == target).sum())
        total += inp.shape[0]
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

def train_out_proj_fast(SNN, args, epochs, batch, snn_samples, targets, cat=False, lr=1e-3, weight_decay=0.0):
    # train output projections from all layers (and no layer)
    losses_out = []
    beta = 1.0
    print_interval = 10*batch
    out_projs = [simple_out(700, 20, beta=beta)]
    optimizers = [torch.optim.AdamW(out_projs[0].parameters(), lr=lr, weight_decay=weight_decay)]
    for lay in range(len(SNN.layers)):
        if cat:
            hiddenshape = 700 + sum(args.n_hidden[:lay+1])
        else:
            hiddenshape = args.n_hidden[lay]
        out_projs.append(simple_out(hiddenshape, 20, beta=beta))
        optimizers.append(torch.optim.AdamW(out_projs[-1].parameters(), lr=lr, weight_decay=weight_decay))
        optimizers[-1].zero_grad()
    SNN.eval()
    acc = []
    correct = (len(SNN.layers) + 1)*[0]
    with torch.no_grad():
        for epoch in trange(epochs):
            shuffled = np.arange(len(snn_samples[0]))
            np.random.shuffle(shuffled)
            snn_samples = [snn_samples[lay][shuffled] for lay in range(len(snn_samples))]
            targets = targets[shuffled]
            for idx in range(0, len(snn_samples[0]), batch):
                until = min(idx + batch, len(snn_samples[0]))
                target = targets[idx:until]
                logit_lists = [None for _ in range(len(SNN.layers)+1)]
                logit_lists[0] = out_projs[0](snn_samples[0][idx:until].float(), target)[1]
                for lay in range(len(SNN.layers)):
                    if cat:
                        datastep = torch.cat([snn_samples[l][idx:until].float() for l in range(lay+2)], dim=-1).float()
                    else:
                        datastep = snn_samples[lay+1][idx:until].float()
                    logit_lists[lay+1] = out_projs[lay+1](datastep, target)[1]
                preds = [logit_lists[lay].argmax(axis=-1) for lay in range(len(SNN.layers)+1)]
                correct = [correct[lay] + (preds[lay] == target).sum() for lay in range(len(SNN.layers)+1)]
                for out_proj in out_projs:
                    out_proj.reset()

                losses_out.append(torch.tensor([torch.nn.functional.cross_entropy(logit_lists[lay], target.squeeze().long()) for lay in range(len(SNN.layers)+1)], requires_grad=False))

                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()
            
            print(f'Cross Entropy Loss: {(torch.stack(losses_out)[-len(snn_samples[0])//batch:].mean(dim=0)).numpy()}\n' +
                        f'Correct: {100*np.array(correct)/len(snn_samples[0])}%')
            acc.append(np.array(correct)/print_interval)
            correct = (len(SNN.layers) + 1)*[0]
    return out_projs, np.asarray(acc), torch.stack(losses_out)

def get_samples(SNN, dataloader, n_hidden, device):
    SNN.eval()
    target = dataloader.y
    samples = dataloader.x
    snn_samples = [torch.zeros(len(dataloader), n, dtype=torch.uint8) for n in n_hidden]
    snn_samples.insert(0, samples.sum(dim=1).squeeze())
    batch = 64
    for idx in trange(len(samples)//batch):
        SNN.reset(0)
        until = min(idx*batch + batch, len(samples))
        sample = samples[idx*batch:until].squeeze()
        for step in range(sample.shape[1]):
            logits, _, _ = SNN(sample[:,step].float().to(device), 0)
            snn_samples
            for lay in range(len(SNN.layers)):
                snn_samples[lay+1][idx*batch:until] += logits[lay].cpu().int().squeeze()
    return snn_samples, target

def train_out_proj_closed_form(args, snn_samples, targets, cat=False, ridge=False):
    # Closed form solution
    from scipy.linalg import lstsq
    from sklearn.linear_model import Ridge
    out_projs = [simple_out(700, 20, beta=1.0)]
    for lay in range(len(args.n_hidden)):
        if cat:
            hiddenshape = 700 + sum(args.n_hidden[:lay+1])
        else:
            hiddenshape = args.n_hidden[lay]
        out_projs.append(simple_out(hiddenshape, 20, beta=1.0))
    # snn_samples = get_samples(SNN, train_loader, args.n_hidden)[0]
    b = torch.nn.functional.one_hot(targets.to(torch.int64), 20).float().numpy()
    for i in range(len(snn_samples)):
        if cat:
            A = torch.cat([*[snn_samples[lay].float() for lay in range(i+1)]], dim=-1).numpy()
        else:
            A = snn_samples[i].float().numpy()
        if ridge:
            clf = Ridge(alpha=1, solver='svd')
            clf.fit(A, b)
            W = clf.coef_
        else:
            W = lstsq(A, b)[0].T

        out_projs[i].out_proj.weight.data = torch.from_numpy(W).float()
        print(W.shape, W.max(), W.min())
    return out_projs