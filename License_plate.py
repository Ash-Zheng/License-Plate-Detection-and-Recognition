import cv2
import numpy as np

# img = cv2.imread("plate/test5.png")
# img = cv2.resize(img, (600, 400))


def preprocessing(img): #图片预处理
    shape = img.shape
    img_Gas = cv2.GaussianBlur(img, (5, 5), 0)
    img_B = cv2.split(img_Gas)[0]
    img_G = cv2.split(img_Gas)[1]
    img_R = cv2.split(img_Gas)[2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_HSV = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2HSV)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (img_HSV[:, :, 1][i, j]>100) and (img_B[i, j] > 70) and (img_R[i,j]<55):
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0



    kernel1 = np.ones((3, 3))
    kernel2 = np.ones((5, 5))
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯平滑
    erosion = cv2.erode(img_gray, kernel1, iterations=1)  # 腐蚀操作，去除边缘噪点
    dilate = cv2.dilate(erosion, kernel1, iterations=5)  # 膨胀操作，使图形更加圆润
    close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel2)  # 封闭运算去除空腔
    ret, img_finish = cv2.threshold(close, 127, 255, cv2.THRESH_BINARY)  # 再次二值化

    cv2.imshow("Image", img_finish)
    cv2.namedWindow("Image")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_finish


def getposition(img, img_finish):
    # 车牌位置获取
    contours, _ = cv2.findContours(img_finish, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    len_max = 0
    pls = 0
    for i in range(len(contours)):
        x_min = np.min(contours[i][ :, :, 0])
        x_max = np.max(contours[i][ :, :, 0])
        y_min = np.min(contours[i][:, :, 1])
        y_max = np.max(contours[i][:, :, 1])
        length = x_max - x_min
        width = y_max - y_min

        if (length/width > 3) and (length/width < 4) and y_min > 100 and x_min > 100 and length > len_max:
            pls = i
            len_max = length

    x_min = np.min(contours[pls][ :, :, 0])
    x_max = np.max(contours[pls][ :, :, 0])
    y_min = np.min(contours[pls][ :, :, 1])
    y_max = np.max(contours[pls][ :, :, 1])
    length = x_max - x_min
    width = y_max - y_min
    rate = length/width

    if rate < 3:
        y_min + 5
        y_max - 5

    dstimg = img[y_min:y_max, x_min:x_max]
    return dstimg, x_min, x_max, y_min, y_max


def getnum(dstimg):
    # 获取黑底白字
    dstimg = cv2.resize(dstimg, (250, 40))
    dst_Gas = cv2.GaussianBlur(dstimg, (5, 5), 0)
    dst_HSV = cv2.cvtColor(dstimg, cv2.COLOR_BGR2HSV)
    dst_B = cv2.split(dstimg)[0]
    dst_G = cv2.split(dstimg)[1]
    dst_R = cv2.split(dstimg)[2]
    dst_gray = cv2.cvtColor(dstimg, cv2.COLOR_BGR2GRAY)
    for i in range(40):
        for j in range(250):
            if (dst_B[i, j] > 120) and (dst_G[i, j] > 80) and (dst_R[i, j] > 80) and i > 3 and j > 3:
                dst_gray[i, j] = 255
            else:
                dst_gray[i, j] = 0

    return dst_gray


# def main():
#     img_finish = preprocessing(img)
#     dstimg = getposition(img_finish)
#     final_picture = getnum(dstimg)  # 最终图片（未进行xcut和ycut）
#     cv2.imshow("Image", final_picture)
#     cv2.namedWindow("Image")
#     # cv2.imshow("Image", img_finish)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()