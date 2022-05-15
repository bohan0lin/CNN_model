import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    train_data_x = pd.read_pickle("~/desktop/brandeis/2022spring/COSI165/AS-2/datasets/train_data_x.pkl")

    train_data_y = pd.read_pickle("~/desktop/brandeis/2022spring/COSI165/AS-2/datasets/train_data_y.pkl")

    test_data_x = pd.read_pickle("~/desktop/brandeis/2022spring/COSI165/AS-2/datasets/test_data_x.pkl")

    test_data_y = pd.read_pickle("~/desktop/brandeis/2022spring/COSI165/AS-2/datasets/test_data_y.pkl")

    return train_data_x, train_data_y, test_data_x, test_data_y

# RCNN model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5, 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(6, 6, 1, 1)
        self.conv3 = nn.Conv2d(6, 6, 1, 1)
        self.conv4 = nn.Conv2d(6, 12, 5, 1)
        
        self.fc1 = nn.Linear(2028, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
    
    
    def forward(self, x):
        
        # first conventional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # second conventional layer
        x = self.conv2(x)
        x = self.relu(x)
        # third conventional layer which has a residual connection with the forth
        residual = x
        x = self.conv3(x)
        x = self.relu(x)
        x += residual
        # forth conventional layer
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        
        # three fully connected layers with activation functions
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x



