import torch
import torchvision
from d2l import torch as d2l
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet34_Weights

from mymodule import MyDatasets

train_data = MyDatasets('train_dataset_5.pickle')

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)

# 微调训练模型
def get_net(device):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
    # 定义一个新的输出网络
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 5))
    # 将模型参数分配给GPU
    finetune_net = finetune_net.to(device)
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


def train(net, train_iter, num_epochs, lr, wd, device):
    net = net.to(device)
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()
        sum_loss = 0.0
        accurate = 0
        for data in train_iter:
            x, y = data
            x = x.to(device=device)
            y = y.to(device=device)
            trainer.zero_grad()
            output = net(x)
            loss_in = loss(output, y).sum()
            loss_in.backward()
            trainer.step()
            sum_loss += loss_in
            accurate += (output.argmax(1) == y).sum().float()

        print('第{}轮训练集的正确率:{:.2f}%,损失:{:.5f}'.format(epoch + 1, accurate / len(train_data) * 100, sum_loss))


devices, num_epochs, lr, wd = d2l.try_gpu(), 100, 1e-3, 1e-4
net = get_net(devices)
train(net, train_dataloader, num_epochs, lr, wd, devices)

# 保存模型
torch.save(net, 'CIFAR_CNN_model.pth')
