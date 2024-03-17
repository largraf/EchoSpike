import torch
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from torchvision.transforms import v2


def augment_nmnist(x):
    transform = v2.Compose([v2.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.3), scale=(0.6, 1.1))])
    return transform(x)

def augment_shd(x):
    x = x.transpose(0, 1)
    transform = v2.RandomAffine(degrees = 0, translate = (0.05, 0.05))
    return transform(x).transpose(0, 1)


class classwise_loader():
    def __init__(self, x, y, num_classes, batch_size=1):
        torch.manual_seed(123)
        if type(x) == torch.Tensor:
            # For regular MNIST & SHD
            self.x = x
        else:
            # For NMNIST
            self.x = None
            self.data = x
        self.batch_size = batch_size
        self.y = torch.tensor(y)
        self.num_classes = num_classes
        self.idx_per_target = num_classes * [0]
        self.target_indeces = [torch.argwhere(self.y == t).squeeze() for t in range(num_classes)]
        self.len = sum([len(ti) for ti in self.target_indeces])
        for target in range(self.num_classes):
            self.shuffle(target)

    
    def __len__(self):
        return self.len    

    def shuffle(self, target):
        idx = torch.randperm(len(self.target_indeces[target]))
        self.target_indeces[target] = self.target_indeces[target][idx]
        self.idx_per_target[target] = 0
    
    def next_item(self, target, contrastive=False):
        if type(target) == int:
            target = [target]
        indeces = []
        for ta in target:
            ta = int(ta)
            if contrastive or ta == -1:
                if ta == -1:
                    ta = torch.randint(0, self.num_classes, (1,)).item()
                else:
                    next_target = torch.randint(0, self.num_classes-1, (1,)).item()
                    if next_target >= ta:
                        next_target += 1
                    ta = next_target
            if self.idx_per_target[ta] >= len(self.target_indeces[ta]):
                self.shuffle(ta)
            indeces.append(self.target_indeces[ta][self.idx_per_target[ta]])
            self.idx_per_target[ta] += 1

        if self.x is None:
            # For NMNIST
            imgs = []
            targets = []
            for i in indeces:
                im, t = self.data[i]
                imgs.append(torch.tensor(im).view(im.shape[0], -1))
                targets.append(t)
            return torch.stack(imgs).transpose(0, 1), torch.tensor(targets)
        return self.x[indeces,].transpose(0, 1), self.y[indeces,]


def load_SHD(batch_size=1):
    # load SHD dataset
    # Note: SHD dataset originally from tonic, but due to a bug on the cluster I had to first download it locally
    shd_train_x = torch.load('./data/SHD/shd_train_x.torch')
    shd_train_y = torch.load('./data/SHD/shd_train_y.torch').squeeze()
    train_loader = classwise_loader(shd_train_x, shd_train_y, 20, batch_size=batch_size)

    shd_test_x = torch.load('./data/SHD/shd_test_x.torch')
    shd_test_y = torch.load('./data/SHD/shd_test_y.torch').squeeze()
    test_loader = classwise_loader(shd_test_x, shd_test_y, 20, batch_size=batch_size)
    return train_loader, test_loader

def load_classwise_NMNIST(n_time_steps, split_train=False, batch_size=1):
    """
    Load the Poisson spike encoded MNIST dataset with the specified parameters.

    Parameters:
        n_time_steps (int): The number of time steps for the spike encoding.
        batch_size (int, optional): The batch size for the data loaders. Default is 1.
        scale (int, optional): The scaling factor for the spike encoding. Default is 1.

    Returns:
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
    """
    import tonic
    from tonic import transforms, DiskCachedDataset
    # load NMNIST dataset
    sensor_size = tonic.datasets.NMNIST.sensor_size
    print(sensor_size)
    transf = [transforms.Denoise(filter_time=10000),
              transforms.ToFrame(sensor_size=sensor_size,
                                 n_time_bins=n_time_steps)]
    frame_transform = transforms.Compose(transf)

    trainset = tonic.datasets.NMNIST(save_to='./data',
                                     transform=frame_transform, train=True, first_saccade_only=True)
    testset = tonic.datasets.NMNIST(save_to='./data',
                                    transform=frame_transform, train=False, first_saccade_only=True)
    trainset_cached = DiskCachedDataset(trainset, cache_path="./data")

    test_loader = classwise_loader(testset, testset.targets, 10, batch_size)

    if split_train == True:
        # Optionally split the train set in a two split
        split = 0.9
        shuffled_idx = torch.randperm(len(trainset))
        indeces_1 = shuffled_idx[:int(split*len(trainset))]
        indeces_2 = shuffled_idx[int(split*len(trainset)):]
        targets_1 = torch.ones_like(shuffled_idx) * -1
        targets_1[indeces_1] = torch.tensor(trainset.targets)[indeces_1]
        train_loader_1 = classwise_loader(trainset_cached, targets_1, 10, batch_size)
        targets_2 = torch.ones_like(shuffled_idx) * -1
        targets_2[indeces_2] = torch.tensor(trainset.targets)[indeces_2]
        train_loader_2 = classwise_loader(trainset_cached, targets_2, 10, batch_size)
        return train_loader_1, train_loader_2, test_loader

    train_loader = classwise_loader(trainset, 10, batch_size)
    return train_loader, test_loader

def load_classwise_PMNIST(n_time_steps, scale=1, split_train=False):
    """
    Load the Poisson spike encoded MNIST dataset with the specified parameters.

    Parameters:
        n_time_steps (int): The number of time steps for the spike encoding.
        batch_size (int, optional): The batch size for the data loaders. Default is 1.
        scale (int, optional): The scaling factor for the spike encoding. Default is 1.

    Returns:
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
    """
    import torchvision
    import torchvision.transforms as transforms
    torch.manual_seed(123)
    trainset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                           download=True)
    testset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                           download=True)
    train_x = snn.spikegen.rate(trainset.data * 2**-8 * scale, n_time_steps).swapaxes(0,1).view(trainset.data.shape[0], n_time_steps, -1)
    test_x = snn.spikegen.rate(testset.data * 2**-8 * scale, n_time_steps).swapaxes(0,1).view(testset.data.shape[0], n_time_steps, -1)
    test_loader = classwise_loader(test_x, testset.targets, 10)

    if split_train == True:
        # Optionally split the train set in an 80/20 split
        split = 0.9
        shuffled_idx = torch.randperm(len(train_x))
        train_x = train_x[shuffled_idx]
        trainset.targets = trainset.targets[shuffled_idx]
        train_loader_1 = classwise_loader(train_x[:int(split*len(train_x))], trainset.targets[:int(split*len(train_x))], 10)
        train_loader_2 = classwise_loader(train_x[int(split*len(train_x)):], trainset.targets[int(split*len(train_x)):], 10)
        return train_loader_1, train_loader_2, test_loader

    train_loader = classwise_loader(train_x, trainset.targets, 10)
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
