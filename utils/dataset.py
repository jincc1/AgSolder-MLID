import torch
from torch.utils.data import Dataset
import numpy as np

class FeatureDataset(Dataset):
    """
    Transforms data from NumPy to Tensor format.
    x is a 2D numpy array: [x_size, x_features]
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])

class AttributeDataset(Dataset):
    """
    Transforms NumPy arrays to a format readable by PyTorch.
    x is a 2D numpy array: [x_size, x_features]
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx]), torch.Tensor(self.y[idx])