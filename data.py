import torch
from torch.utils.data import DataLoader, TensorDataset


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

def load_SHD(n_time_bins, batch_size=1):
    # load SHD dataset
    shd_train_x = torch.load('./data/SHD/shd_train_x.torch')
    shd_train_y = torch.load('./data/SHD/shd_train_y.torch')
    trainset = TensorDataset(shd_train_x, shd_train_y)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    shd_test_x = torch.load('./data/SHD/shd_test_x.torch')
    shd_test_y = torch.load('./data/SHD/shd_test_y.torch')
    testset = TensorDataset(shd_test_x, shd_test_y)
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
