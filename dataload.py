import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# 继承Dataset类，读取自己的训练数据
class ProvinceDataset(Dataset):
    # root为图像根目录
    def __init__(self, root, transform=None):
        a = os.listdir(root)  # 获取子目录
        for i in a:
            if i == '.DS_Store':  # 剔除MacOS系统中的文件夹显示配置文件
                a.remove(i)
        files = []
        for i in a:
            temp = np.array([x.path for x in os.scandir(root + '/' + i)
                             if x.name.endswith(".bmp") or x.name.endswith(".png")])
            files = np.append(files, temp)

        self.image_files = files
        self.transform = transform


    def __getitem__(self, index):
        # 读取图像返回图像和标签
        img_path = self.image_files[index]
        img = cv2.imread(self.image_files[index], 0)  # 单通道读取
        # 图像二值化
        for i in range(40):
            for j in range(32):
                if img[i, j] > 0:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
        img = [img]  # 增加一个纬度，因为CNN输入必须为四维张量
        img = torch.Tensor(img)
        path_list = img_path.split('/')
        label = int(path_list[-2])
        return img, label

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)


class NumberDataset(Dataset):
    # root为图像根目录
    def __init__(self, root, transform=None):
        a = os.listdir(root)  # 获取子目录
        for i in a:
            if i == '.DS_Store':  # 剔除MacOS系统中的文件夹显示配置文件
                a.remove(i)
        files = []
        for i in a:
            temp = np.array([x.path for x in os.scandir(root + '/' + i)
                             if x.name.endswith(".bmp") or x.name.endswith(".png")])
            files = np.append(files, temp)

        self.image_files = files
        self.transform = transform


    def __getitem__(self, index):
        # 读取图像返回图像和标签
        img_path = self.image_files[index]
        img = cv2.imread(self.image_files[index], 0)  # 单通道读取
        # 图像二值化
        for i in range(40):
            for j in range(32):
                if img[i, j] > 0:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
        img = [img]  # 增加一个纬度，因为CNN输入必须为四维张量
        img = torch.Tensor(img)
        path_list = img_path.split('/')
        label = int(path_list[-2])
        labelList = torch.zeros(1)
        labelList[0] = label
        return img, label

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)

