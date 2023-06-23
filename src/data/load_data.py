import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.signal_utils import PickleLoad

def Load_CIFAR10_Data(data_path:str):
    """
    Loads all fo the pickle files of the CIFA-10 dataset
    """

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #Load in the batches
    for ii in range(5):
        batch = PickleLoad(os.path.join(data_path, 'data_batch_{}'.format(ii+1)))

        X = np.concatenate((X, batch[b'data'])) if ii > 0 else batch[b'data']
        y = np.concatenate((y, batch[b'labels'])) if ii > 0 else batch[b'labels']
        filenames = np.concatenate((filenames, batch[b'filenames'])) if ii > 0 else batch[b'filenames']


    #Reshape the arrays
    X_reshape = np.reshape(X,(X.shape[0], 3, int(np.sqrt(X.shape[-1]/3)), int(np.sqrt(X.shape[-1]/3))))

    #ShowImage(X_reshape[2])

    test_batch = PickleLoad(os.path.join(data_path, 'test_batch'))

    X_test = np.array(test_batch[b'data'])
    X_test_reshape = np.reshape(X_test,(X_test.shape[0], 3, int(np.sqrt(X_test.shape[-1]/3)), int(np.sqrt(X_test.shape[-1]/3))))
    y_test = np.array(test_batch[b'labels'])

    return X_reshape, y, X_test_reshape, y_test, classes


def ShowImage(image_array: np.ndarray):
    """
    Shows the image for the array passed (3,M,N)
    """

    corrected_image = np.swapaxes(image_array.transpose(),0,1)

    plt.imshow(corrected_image)
    plt.show()


if __name__ == '__main__':

    CIFAR_10_path = os.path.join(os.getcwd(), 'App', 'data', 'cifar-10')

    Load_CIFAR10_Data(CIFAR_10_path)