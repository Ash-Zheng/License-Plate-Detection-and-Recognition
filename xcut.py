# 去除y轴上下噪声


def x_cut(img):
    x_value = []
    y_pls = []
    # 图像预处理
    for i in range(40):  # 列
        inter = 0  # 沿y轴的积分值
        for j in range(250):  # 行
            inter += img[i, j]
        x_value.append(inter)
        y_pls.append(i)

    # plt.plot(y_pls, x_value, 'r')
    # plt.show()
    check = 8000
    cnt = 0
    for i in range(39):  # 检测上下是否有噪声
        if (x_value[i] <= check) and (x_value[i + 1] >= check):
            cnt += 1
        elif (x_value[i] >= check) and (x_value[i + 1] <= check):
            cnt += 1
    x1 = 0
    x2 = 0
    if cnt >= 6:  # 有噪声干扰
        x_tag = []
        for i in range(39):  # 检测上下是否有噪声
            if (x_value[i] < check) and (x_value[i + 1] >= check):
                x_tag.append(i)
            elif (x_value[i] >= check) and (x_value[i + 1] <= check):
                x_tag.append(i)
        x1 = x_tag[2]
        x2 = x_tag[3]

        while x_value[x1 - 1] <= x_value[x1]:
            x1 -= 1
        while x_value[x2] >= x_value[x2 + 1]:
            x2 += 1
    elif cnt >= 3:
        x_tag = []
        for i in range(39):  # 检测上下是否有噪声
            if (x_value[i] < check) and (x_value[i + 1] >= check):
                x_tag.append(i)
            elif (x_value[i] >= check) and (x_value[i + 1] <= check):
                x_tag.append(i)
        x1 = 0
        x2 = x_tag[1]
        while x_value[x1] < 1000:
            x1 += 1
        while x_value[x2] >= x_value[x2 + 1]:
            x2 += 1
    else:  # 没有噪声干扰
        x1 = 0
        x2 = 39
        while x_value[x1 + 1] <= x_value[x1]:
            x1 += 1
        while x_value[x2 - 1] <= x_value[x2]:
            x2 -= 1

    subimg = img[x1:x2, 0:250]  # y轴划分完的图片
    return subimg

