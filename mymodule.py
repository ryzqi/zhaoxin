import pickle
import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision.models import ResNet50_Weights
import torchvision

class MyDatasets(Dataset):
    def __init__(self, data_route):
        with open(data_route, 'rb') as file:
            data_dict = pickle.load(file, encoding='iso-8859-1')
        raw_x = data_dict['x']
        raw_y = data_dict['y']
        self.data_x = raw_x.reshape((raw_x.shape[0], 3, 32, 32))
        self.data_y = raw_y.reshape((raw_y.shape[0]))

    def __getitem__(self, index):
        x = self.data_x[index, :, :, :]
        y = self.data_y[index]
        y = int(y)
        return torch.Tensor(x), y

    def __len__(self):
        return self.data_x.shape[0]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = self.get_net()

    def get_net(self):
        finetune_net = nn.Sequential()
        finetune_net.features = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 定义一个新的输出网络
        finetune_net.output_new = nn.Sequential(nn.Linear(1000, 512),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(),
                                                nn.Linear(512, 128),
                                                nn.BatchNorm1d(128),
                                                nn.ReLU(),
                                                nn.Linear(128, 5))

        for i in range(0, 7, 3):
            nn.init.xavier_uniform_(finetune_net.output_new[i].weight);
        # 冻结参数
        for param in finetune_net.features.parameters():
            param.requires_grad = False
        return finetune_net

    def forward(self, x):
        x = self.model(x)
        return x
