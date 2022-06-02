import csv

from tqdm import tqdm

from dataset import get_test_loader
from model import *

train_path = './data/covid.train.csv'
test_path = './data/covid.test.csv'


def save_preds(datas, path):
    '''Save the data into csv'''
    with open(path, 'w', newline='') as fp:  # define there is no newline between two rows
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for index, data in enumerate(datas):
            writer.writerow([index, data])


def test(test_loader):
    '''Predict the future outcome'''
    model = My_Model()
    model.load_state_dict(torch.load('model.ckpt'))  # loading my training model
    result = []

    with torch.no_grad():
        for x in tqdm(test_loader):
            y = model(x)  # y is a tensor, whose size is [2,1]  (2 is row, 1 is column)
            y = y.numpy().tolist()  # y becomes a list, which only contains the outcome
            result += [i[0] for i in y]

    save_preds(result, 'outcome.csv')
    print('-' * 100)
    print('managed to save file')


if __name__ == '__main__':
    test_loader = get_test_loader(test_path=test_path, mode='test', batch_size=128)
    test(test_loader)
