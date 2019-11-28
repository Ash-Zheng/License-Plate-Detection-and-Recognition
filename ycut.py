# 字符分割

def y_cut(img):
    shape = img.shape
    x_value = []
    y_pls = []
    # 图像预处理
    for i in range(shape[1]):  # 行
        inter = 0  # 沿y轴的积分值
        for j in range(shape[0]):  # 列
            inter += img[j, i]
        x_value.append(inter)
        y_pls.append(i)
        # 获取投影到x轴上的积分值

    divide = []  # 分割坐标列表
    start = 20
    x0 = start
    x1 = start
    while x_value[x0] > 0:
        x0 -= 1
    while x_value[x1] > 0:
        x1 += 1

    if (x1 - x0) < 21:
        while x_value[x0] <= 0:
            x0 -= 1
        while x_value[x0] > 0:
            x0 -= 1

    divide.append([x0, x1])

    for i in range(7):
        x0 = x1
        while x_value[x0] <= 0:
            x0 += 1
        x1 = x0
        while x_value[x1] > 0:
            x1 += 1

        if i != 1:
            divide.append([x0, x1])

    for item in divide:
        if item[1] - item[0] < 15:
            mid = int((item[1] + item[0]) / 2)
            item[1] = mid + 12
            item[0] = mid - 12

    return divide

