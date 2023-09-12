import os

import torch

link = 'https://github.com/ryzqi/zhaoxin.git'
def test(test_x):
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
    test_x = torch.from_numpy(test_x).float() / 255
    model = torch.load(os.path.split(os.path.realpath(__file__))[0] + '/CIFAR_CNN_model.pth')
    model.eval()
    output = model(test_x)
    result = torch.argmax(output, dim=1)
    return result.numpy()