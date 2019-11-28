import os
import cv2
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from modeling import NetP
from dataload import ProvinceDataset

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 归一化处理，数据数值转化为[-1，1]之间的数
transform = transforms.Compose(
    [transforms.TenCrop((32, 40)),
     transforms.RandomRotation(20),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transforms用于预处理

trainset = ProvinceDataset(root="data/train_images/training-set/chinese", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)  # 批处理读取数据集

validationset = ProvinceDataset(root="data/train_images/validation-set/chinese", transform=transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=True, num_workers=0)  # 批处理读取测试数据集

classes = ('京', '沪', '苏')


# 测试数据集是否读取成功
print(len(trainset))
print(len(trainloader))
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)

net = NetP()  # 创建一个神经网络

lossfunc = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数
# 采用Stochastic gradient descent的方法更新参数，learning rate为0.001，动量设为0.9
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

# 训练神经网络
for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0  # 清空loss
    loss_value = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # 获取数据和标签值，一批4个
        optimizer.zero_grad()  # 清除参数的梯度，每次循环重新计算

        # forward + backward + optimize
        outputs = net(inputs)
        loss = lossfunc(outputs, labels)
        loss.backward()  # 对损失函数求偏导
        optimizer.step()  # 更新参数

        # print statistics
        running_loss += loss.item()   # 用于输出损失值
        loss_value = loss.item()

        if i % 5 == 4:    # 每1000次迭代输出一次损失值
            print('第%d次遍历数据集，第%d次迭代，损失值：%.5f' % (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

    # if loss_value < 0.00001:
    #     break

print('Finished Training')

torch.save(net.state_dict(), 'model/provinceweight.pt')
torch.save(net, 'model/provincenet.pt')
