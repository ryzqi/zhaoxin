import pickle
import torch
from torch.utils.data import Dataset


class MyDatasets(Dataset):
    def __init__(self, data_route):
        with open(data_route, 'rb') as file:
            data_dict = pickle.load(file, encoding='iso-8859-1')
        raw_x = data_dict['x']
        raw_y = data_dict['y']
        self.data_x = raw_x.reshape((raw_x.shape[0], 3, 32, 32))
        self.data_y = raw_y.reshape((raw_y.shape[0]))

    def __getitem__(self, index):
        x = self.data_x[index,:,:,:]
        y = self.data_y[index]
        x = torch.from_numpy(x).float() / 255
        y = int(y)
        return x, y

    def __len__(self):
        return self.data_x.shape[0]



