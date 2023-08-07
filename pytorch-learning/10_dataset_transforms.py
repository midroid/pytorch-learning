from typing import Any
from torch.utils.data import Dataset
import torch
import torchvision
import numpy as np

dataset = torchvision.datasets.MNIST(
    root='./data',
    transform=torchvision.transforms.ToTensor(),
    # download=True
)

class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # We do not convert tensors here
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform


    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample


    def __len__(self):
        return self.n_samples
    

class ToTensor:
    def __call__(self, sample) -> Any:
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target
    

# dataset = WineDataset(transform=ToTensor())
dataset = WineDataset(transform=None)
# dataset = WineDataset(transform=MulTransform)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
