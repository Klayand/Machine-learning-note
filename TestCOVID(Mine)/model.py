import torch
import os
import torch.nn as nn
from dataset import get_loader


train_path = './data/covid.train.csv'
test_path = './data/covid.test.csv'


class My_Model(nn.Module):
    '''
    Define my neural network:
    three layers or more???
    activate function ReLU
    how to get the input dim??
    '''
    def __init__(self, input_dim=116):
        super(My_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        '''
        the calculate part: call the model function
        '''
        return self.model(x)



