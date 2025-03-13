import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Here, we reimplement the model from Zippelius et al.: Predicting thermal resistance of solder joints based 
on Scanning Acoustic Microscopy using Artificial Neural Networks(https://ieeexplore.ieee.org/document/9939465).
'''

class SAM_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SAM_CNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) 
        x = self.pool2(F.relu(self.conv2(x)))  
        x = self.pool3(F.relu(self.conv3(x)))  

        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        
        return x