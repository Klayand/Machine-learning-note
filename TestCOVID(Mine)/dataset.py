import torch
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import numpy as np

train_path = './data/covid.train.csv'
test_path = './data/covid.test.csv'


class My_Dataset(Dataset):
    '''Define my own dataset'''
    def __init__(self, path, mode):
        self.mode = mode
        self.data = read_data(path)
        if mode is not 'test':
            self.train_set, self.valid_set = data_split(self.data, mode)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_set)
        if self.mode == 'valid':
            return len(self.valid_set)
        if self.mode == 'test':
            return len(self.data)

    def __getitem__(self, item):
        '''
        x : Features
        y : Labels
        :return: x,y
        '''
        if self.mode == 'train':
            now = self.train_set[item]
            self.x = now[1:-1]
            self.y = now[-1]
            return torch.tensor(self.x), torch.tensor(self.y)

        if self.mode == 'valid':
            now = self.valid_set[item]
            self.x = now[1:-1]
            self.y = now[-1]
            return torch.tensor(self.x), torch.tensor(self.y)

        else:
            now = self.data[item]
            return torch.tensor(now[1:])


def read_data(path):
    '''Get data from the csv files'''
    datas = []
    with open(path) as f:
        f_csv = csv.reader(f)
        for index,data in enumerate(f_csv):
            # Get the datas except the first column
            # the first row is id
            if index != 0:
                datas.append([float(i) for i in data])

    return datas


def data_split(dataset, mode):
    '''Split provided training data into training set and validation set'''
    valid_ratio = 0.8
    train_set_size = int(len(dataset) * valid_ratio)
    valid_set_size = len(dataset) - train_set_size
    # print(train_set_size,valid_set_size)  # define the size of the train_set and valid_set

    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size])
    # train_set and valid_set is list objects.
    return train_set, valid_set


def get_loader(train_path, mode, batch_size=128):
    '''
    TODO: to get the train or valid dataloader
    '''

    set = My_Dataset(path=train_path, mode=mode)
    train_loader = DataLoader(set, batch_size=batch_size, shuffle=True)
    return train_loader


def get_test_loader(test_path, mode ='test', batch_size=128):
    '''
    TODO: to get the test dataloader
    '''

    set = My_Dataset(path=test_path, mode=mode)
    test_loader = DataLoader(set, batch_size=batch_size, shuffle=False)
    return test_loader


if __name__ == '__main__':
    data_set = My_Dataset(path=train_path,mode='train')









