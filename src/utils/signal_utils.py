import torch
from torch import tensor
from torch.utils.data import Dataset
import numpy as np
import pickle

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = np.zeros((len(y),10))
        for ii, index in enumerate(y):
            self.y[ii, int(index)] = 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx,:,:,:]
        y = self.y[idx,:]
        return tensor(X, dtype=torch.float32), tensor(y, dtype=torch.float32)


def PickleDump(obj, filepath):
    """
    Pickles and object at a given filepath
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def PickleLoad(filepath):
    """
    Loads an object from a pickle filepath 
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f, encoding='bytes')
    return obj