import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def LoadCIFAR10(image_size:int=32,
                data_dir:str = os.path.join(os.path.dirname(os.getcwd()), 'data')):
    # CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)

    return dataset.data, dataset.targets, dataset.classes

def ShowImage(image_array: np.ndarray):
    """
    Shows the image for the array passed (3,M,N)
    """

    corrected_image = np.swapaxes(image_array.transpose(),0,1)

    plt.imshow(corrected_image)
    plt.show()
