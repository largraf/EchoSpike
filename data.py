import torch
from torch.utils.data import DataLoader, TensorDataset
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

class classwise_loader():
    def __init__(self, x, y, num_classes):
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.idx_per_target = num_classes * [0]
        self.target_indeces = [torch.argwhere(self.y == t).squeeze() for t in range(num_classes)]
        for target in range(self.num_classes):
            self.shuffle(target)

    def __len__(self):
        return len(self.x)
    
    def shuffle(self, target):
        idx = torch.randperm(len(self.target_indeces[target]))
        self.target_indeces[target] = self.target_indeces[target][idx]
        self.idx_per_target[target] = 0
    
    def next_item(self, target: int, contrastive=False):
        if contrastive:
            if target == -1:
                target = torch.randint(0, self.num_classes, (1,)).item()
            else:
                next_target = torch.randint(0, self.num_classes-1, (1,)).item()
                if next_target >= target:
                    next_target += 1
                target = next_target
        if self.idx_per_target[target] >= len(self.target_indeces[target]):
            self.shuffle(target)
        idx = self.target_indeces[target][self.idx_per_target[target]]
        self.idx_per_target[target] += 1
        return self.x[idx], self.y[idx]


def load_SHD(n_time_bins, batch_size=1):
    # load SHD dataset
    shd_train_x = torch.load('./data/SHD/shd_train_x.torch')
    shd_train_y = torch.load('./data/SHD/shd_train_y.torch').squeeze()
    trainset = TensorDataset(shd_train_x, shd_train_y)
    # train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    train_loader = classwise_loader(shd_train_x, shd_train_y, 20)

    shd_test_x = torch.load('./data/SHD/shd_test_x.torch')
    shd_test_y = torch.load('./data/SHD/shd_test_y.torch').squeeze()
    testset = TensorDataset(shd_test_x, shd_test_y)
    # test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    test_loader = classwise_loader(shd_test_x, shd_test_y, 20)
    return train_loader, test_loader

def load_classwise_PMNIST(n_time_steps, batch_size=1, scale=1):
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
    trainset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                           download=True)
    testset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                           download=True)
    train_x = snn.spikegen.rate(trainset.data * 2**-8 * scale, n_time_steps).swapaxes(0,1).view(trainset.data.shape[0], n_time_steps, -1)
    test_x = snn.spikegen.rate(testset.data * 2**-8 * scale, n_time_steps).swapaxes(0,1).view(testset.data.shape[0], n_time_steps, -1)
    train_loader = classwise_loader(train_x, trainset.targets, 10)
    test_loader = classwise_loader(test_x, testset.targets, 10)
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
