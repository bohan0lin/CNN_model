import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    train_data_x = pd.read_pickle("~/Desktop/Brandeis/2022spring/COSI165/AS-2/datasets/train_data_x.pkl")

    train_data_y = pd.read_pickle("~/Desktop/Brandeis/2022spring/COSI165/AS-2/datasets/train_data_y.pkl")

    test_data_x = pd.read_pickle("~/Desktop/Brandeis/2022spring/COSI165/AS-2/datasets/test_data_x.pkl")

    test_data_y = pd.read_pickle("~/Desktop/Brandeis/2022spring/COSI165/AS-2/datasets/test_data_y.pkl")

    return train_data_x, train_data_y, test_data_x, test_data_y

# CNN model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 12, 5)
        self.fc1 = torch.nn.Linear(12 * 6 * 6, 120)
        self.fc2 = torch.nn.Linear(120, 64)
        self.fc3 = torch.nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features








