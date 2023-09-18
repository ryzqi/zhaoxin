import torch
import os
import torchvision

link = 'https://github.com/ryzqi/zhaoxin.git'


def test(test_x):
    test_x = test_x.reshape((test_x.shape[0], 3, 32, 32))
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               # 标准化图像的每个通道
                                               torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                [0.229, 0.224, 0.225])])
    test_x = transform(test_x)
    test_x = torch.from_numpy(test_x).float() / 255
    model = torch.load(os.path.split(os.path.realpath(__file__))[0] + '/CIFAR_CNN_model.pth')
    model.eval()
    output = model(test_x)
    result = torch.argmax(output, dim=1)
    return result.numpy()
