import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from alive_progress import alive_bar
import os
import sys
from joblib import dump, load
import cv2

from src.utils.signal_utils import CustomDataset


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
        x = torch.swapaxes(x, 1, 3)
        x = torch.swapaxes(x, 2, 3)
        # Pass input through convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from the convolutional layers
        x = torch.flatten(x, start_dim=1)        
        # Pass through the linear layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return self.softmax(x)


class ImageClassifier:
    def __init__(self, save_dir:str, 
                 classes:list=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
                 image_size=32) -> None:
        self.model = MyCNN(classes=classes)
        self.observed_image = ''
        self.model_dir = os.path.join(save_dir, 'CNN')
        self.image_size = image_size

    def trainCNN(self, X, y, num_epochs=50,
                 batch_size=512, lr=0.01):
        """
        Runs through the training and passes back a trained model
        """

        #Set loss function, optimizer, and number of epochs
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        dataset = CustomDataset(X, y)
        dataloader = DataLoader(dataset, shuffle=True, batch_size = batch_size, num_workers=2)

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
        
        dump(self.model, os.path.join(self.model_dir, 'cls_model.joblib'), compress=3)


    def loadModel(self):
        """
        Loads the CNN
        """
        self.model = load(os.path.join(self.model_dir, 'cls_model.joblib'))

    def classify_img(self, img):
        """
        img: n_width x n_height x n_channels
        """
        res_img = cv2.resize(img, dsize=(self.image_size, self.image_size))
        X = torch.tensor(res_img, dtype=torch.float)
        X = torch.unsqueeze(X, 0)
        prediction = self.model(X)
        class_idx = torch.argmax(self.model(X)).detach().cpu().numpy()
        return self.model.classes[class_idx]
        

    

