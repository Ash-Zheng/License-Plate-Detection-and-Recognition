import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from modeling import Net
from modeling import NetN
from dataload import NumberDataset


# 归一化处理，数据数值转化为[-1，1]之间的数
transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transforms用于预处理

trainset = NumberDataset(root="data/train_images/training-set/num_letter", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)  # 批处理读取数据集

validationset = NumberDataset(root="data/train_images/validation-set/num_letter", transform=transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=4, shuffle=True, num_workers=0)  # 批处理读取测试数据集

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z')

# 测试数据集是否读取成功
print(len(trainset))
print(len(trainloader))
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)


net = NetN()  # 创建一个神经网络
lossfunc = nn.CrossEntropyLoss()  # 使用交叉熵作为损失函数
# 采用Stochastic gradient descent的方法更新参数，learning rate为0.001，动量设为0.9
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
# alpha = 0.9
# input = torch.randn(1, 1, 40, 32)
# label = torch.randn(1)
# y1 = torch.zeros(34)
# y2 = torch.zeros(34)
# 训练神经网络
for epoch in range(60):  # loop over the dataset multiple times

    running_loss = 0.0  # 清空loss
    loss_value = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # 获取数据和标签值，一批4个
        # y1 = labels[0]
        # x1 = inputs[0]
        # x2 = inputs[1]
        # y1 = labels[0]
        # y2 = labels[1]
        # lam = np.random.beta(alpha, alpha)
        #
        # input = lam * x1 + (1. - lam) * x2
        # label = lam * y1 + (1. - lam) * y2
        optimizer.zero_grad()  # 清除参数的梯度，每次循环重新计算

        # input = input.unsqueeze(0)
        # label = label.unsqueeze(0)
        # forward + backward + optimize
        outputs = net(inputs)
        loss = lossfunc(outputs, labels)
        loss.backward()  # 对损失函数求偏导
        optimizer.step()  # 更新参数

        # print statistics
        running_loss += loss.item()   # 用于输出损失值
        loss_value = loss.item()

        if i % 50 == 49:    # 每1000次迭代输出一次损失值
            print('第%d次遍历数据集，第%d次迭代，损失值：%.5f' % (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0


print('Finished Training')

# 保存训练好的模型
# torch.save(net.state_dict(), 'model/number_letter_weight.pt')
torch.save(net, 'model/newtrain/num1.pt')


# 小规模测试
# dataiter = iter(validationloader)
# images, labels = dataiter.next()

# outputs = net(images)  # 获得预测值
# maxvalue, predicted = torch.max(outputs, 1)  # 返回最大值并返回其索引
# print('识别结果:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# # print images
# print('正确结果:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# b = images.numpy()
# gray = np.concatenate((b[0][0], b[1][0], b[2][0], b[3][0]), axis=1)

# cv2.imshow("Image", gray)
# cv2.namedWindow("Image")
# # cv2.imshow("Image", img_finish)
# cv2.waitKey(0)
# cv2.destroyAllWindows()