import os
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from modeling import Net
from modeling import NetN
from modeling import NetP
from License_plate import *
import xcut
import ycut


name = "test2.png"

img = cv2.imread("plate/"+name)
img = cv2.resize(img, (600, 400))

# 图片处理部分
img_finish = preprocessing(img)
dstimg, x1, x2, y1, y2 = getposition(img, img_finish)
final_picture = getnum(dstimg)  # 提取到车牌
x_cutted_img = xcut.x_cut(final_picture)
ycut_divide = ycut.y_cut(x_cutted_img)


# print(ycut_divide)
shape = x_cutted_img.shape


classes1 = ('京', '沪', '苏')
classes2 = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z')
result = ''  # 结果保存
net1 = NetP()  # 创建一个神经网络
net1 = torch.load('model/newtrain/provincenet.pt')  # 加载训练好的模型
net1.eval()

net = NetN()  # 创建一个神经网络
net = torch.load('model/newtrain/num.pt')
net.eval()

for i in range(7):
    axis = ycut_divide[i]
    subimg = x_cutted_img[0:shape[0], axis[0]:axis[1]]  # y轴划分完的图片
    # 边缘填充
    top = int(0.1 * subimg.shape[1])
    bottom = int(0.1 * subimg.shape[1])
    left = int(0.1 * subimg.shape[0])
    right = int(0.1 * subimg.shape[0])
    testimg = cv2.copyMakeBorder(subimg, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    testimg = cv2.resize(testimg, (32, 40))

    PIL_img = Image.fromarray(testimg)
    tensor_img = transforms.ToTensor()(PIL_img)
    tensor_img = tensor_img.expand(tensor_img.shape[0], 1, tensor_img.shape[1], tensor_img.shape[2])
    if i == 2:
        result = result + '-'
    if i == 0:
        output = net1(tensor_img)
        _, predict = torch.max(output, 1)
        result = result + classes1[predict]
    else:
        output = net(tensor_img)
        _, predict = torch.max(output, 1)
        result = result + classes2[predict]

    # cv2.imwrite('newnum/'+str(i)+'.png', testimg)
    # cv2.imshow("Image", testimg)
    # cv2.namedWindow("Image")
    # # cv2.imshow("Image", img_finish)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


print(result)
p_center = (x1-100, y1-70)
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 5)
# cv2.putText(img, result, p_center, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
cv2.imshow("Image", img)
cv2.namedWindow("Image")
cv2.waitKey(0)
cv2.destroyAllWindows()