import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from modeling import Net
from dataload import ProvinceDataset

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 归一化处理，数据数值转化为[-1，1]之间的数
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transforms用于预处理

validationset = ProvinceDataset(root="data/train_images/validation-set/num_letter", transform=transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=4, shuffle=True, num_workers=0)  # 批处理读取测试数据集

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z')


# ###################################数据集读取成功，开始构建CNN######################################
net = Net()  # 创建一个神经网络

net = torch.load('model/number_letter_net.pt')


# 测试训练好的模型的检测成功率
class_correct = list(0. for i in range(34))
class_total = list(0. for i in range(34))
with torch.no_grad():
    for data in validationloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        a = labels.size()
        for i in range(a[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(34):
    print('识别准确率-%s：%2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# #小规模测试
# dataiter = iter(validationloader)
# images, labels = dataiter.next()
#
#
# outputs = net(images)  # 获得预测值
# maxvalue, predicted = torch.max(outputs, 1)  # 返回最大值并返回其索引
# print('识别结果:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# # print images
# print('正确结果:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# b = images.numpy()
# gray = np.concatenate((b[0][0], b[1][0], b[2][0], b[3][0]), axis=1)
#
# cv2.imshow("Image", gray)
# cv2.namedWindow("Image")
# # cv2.imshow("Image", img_finish)
# cv2.waitKey(0)
# cv2.destroyAllWindows()