import torch.nn as nn
import torch.nn.functional as func


# 定义神经网络  输入像素为32*40
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 7, 180)
        self.fc2 = nn.Linear(180, 120)
        self.fc3 = nn.Linear(120, 34)
        self.droplayer = nn.Dropout()
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 24 * 5 * 7)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义神经网络  输入像素为32*40
class NetP(nn.Module):
    def __init__(self):
        super(NetP, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.droplayer = nn.Dropout()
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 7)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetN(nn.Module):
    def __init__(self):
        super(NetN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 7, 360)
        self.fc2 = nn.Linear(360, 200)
        self.fc3 = nn.Linear(200, 34)
        self.droplayer = nn.Dropout()
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        x = x.view(-1, 24 * 5 * 7)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x