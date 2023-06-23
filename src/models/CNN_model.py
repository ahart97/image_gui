import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from alive_progress import alive_bar
import os
import sys

from src.data.load_data import Load_CIFAR10_Data
from src.utils.signal_utils import CustomDataset, PickleDump, PickleLoad


class MyCNN(nn.Module):
    def __init__(self, classes):
        super(MyCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Linear layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

        self.classes = classes
        
    def forward(self, x):
        # Pass input through convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Pass through the linear layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return self.softmax(x)


class ImageClassifier:
    def __init__(self, model: MyCNN) -> None:
        self.model = model
        self.observed_image = ''

    def trainCNN(self, X, y):
        """
        Runs through the training and passes back a trained model
        """

        #Set loss function, optimizer, and number of epochs
        lr = 0.01
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        num_epochs = 50

        dataset = CustomDataset(X, y)
        dataloader = DataLoader(dataset, shuffle=True, batch_size = 1000, num_workers=2)

        print('Training model...')
        with alive_bar(num_epochs, force_tty=True) as bar:
            for epoch in range(0, num_epochs, 1):
                for jj, data in enumerate(dataloader):
                    inputs, targets = data

                    #Resets the optimizer to zero grad
                    optimizer.zero_grad()

                    output = self.model(inputs)
                    loss = loss_fn(output, targets)

                    #Back propagate based on the loss
                    loss.backward()

                    #Update coefficients based on the back prop
                    optimizer.step()
                bar()

    def SaveModel(self, save_path):
        """
        Saves the CNN
        """
        PickleDump(self.model, os.path.join(save_path, 'CNN_model.pickle'))

    def LoadModel(self, load_path):
        """
        Loads the CNN
        """
        self.model = PickleLoad(os.path.join(load_path, 'CNN_model.pickle'))

    def classify_img(self, img):
        """
        img: n_width x n_height x n_channels
        """
        X = tensor(img, dtype=torch.float32)
        self.observed_image = self.classes[torch.argmax(self.forward(X)).detach().cpu().numpy()]
        


if __name__ == '__main__':
    """
    Test the CNN model traing and structure
    """

    X, y, X_test, y_test, classes = Load_CIFAR10_Data(os.path.join(os.getcwd(), 'App', 'data', 'cifar-10'))

    trialCNN = MyCNN(classes)
    imgClassifier = ImageClassifier(trialCNN)

    imgClassifier.trainCNN(X, y)
    imgClassifier.SaveModel(os.path.join(os.getcwd(), 'App', 'Models', 'CNNModelData'))


    

