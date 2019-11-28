import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from modeling import NetP
from dataload import ProvinceDataset


# 归一化处理，数据数值转化为[-1，1]之间的数
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transforms用于预处理

trainset = ProvinceDataset(root="data/train_images/training-set/chinese", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)  # 批处理读取数据集

validationset = ProvinceDataset(root="data/train_images/validation-set/chinese", transform=transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=4, shuffle=True, num_workers=0)  # 批处理读取测试数据集

classes = ('京', '闽', '粤', '苏', '沪', '浙')


net = NetP()  # 创建一个神经网络

net = torch.load('model/provincenet.pt')

# 测试训练好的模型的检测成功率
class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))
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
for i in range(6):
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