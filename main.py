import numpy as np
import cv2
from sklearn import metrics
import glob
from matplotlib import pyplot as plt
import random
import math

'''
这个函数是在将图片合为视频时会用到，作用是将图片变为同等大小
'''


def resize(img_array, align_mode):
    _height = len(img_array[0])
    _width = len(img_array[0][0])
    for i in range(1, len(img_array)):
        img = img_array[i]
        height = len(img)
        width = len(img[0])
        if align_mode == 'smallest':
            if height < _height:
                _height = height
            if width < _width:
                _width = width
        else:
            if height > _height:
                _height = height
            if width > _width:
                _width = width

    for i in range(0, len(img_array)):
        img1 = cv2.resize(img_array[i], (_width, _height), interpolation=cv2.INTER_CUBIC)
        img_array[i] = img1

    return img_array, (_width, _height)


'''
此函数的作用是：在文件夹中取出处理后的图片，并将图片合为视频
'''


def images_to_video(path):
    img_array = []

    for filename in glob.glob(path + '/*.png'):
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    # 图片的大小需要一致
    img_array, size = resize(img_array, 'largest')
    fps = 24
    out = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


'''
此函数的作用是：在文件夹中读取出待处理图片，并用模拟退火算法进行图像匹配
'''


def readImg(path):
    template = cv2.imread("00.png", 0)
    for filename in glob.glob(path + '/*.png'):
        ss = filename[-8:]
        target = cv2.imread(filename, 0)
        if target is None:
            print(filename + " is error!")
            continue
        find(target, template, ss)


def find(target, template, ss):
    # 获取目标图片的大小
    target_x, target_y = target.shape
    # 获取模板图片的大小
    template_x, template_y = template.shape
    # 计算候选框的数量
    candidate_x = target_x - template_x
    # candidate_x = target_x - template_x + 1
    candidate_y = target_y - template_y
    # candidate_y = target_y - template_y + 1

    # 1.计算模板图片的概率分布
    print('1.计算模板图片的概率分布和信息熵')
    distribution_template = np.zeros(256, np.int32)  # 记录概率分布
    print('1.1计算模板图片的概率分布')
    for i in range(template_x):
        for j in range(template_y):
            # 统计每个灰度值的数量
            distribution_template[template[i, j]] = distribution_template[template[i, j]] + 1

    startX = 0
    endX = candidate_x

    startY = 0
    endY = candidate_y

    X = random.randint(startX, endX)
    Y = random.randint(startY, endY)

    Z = targetImg(distribution_template, X, Y, target, template_x, template_y)

    '''
    进行模拟退火算法，初始温度为T 1000,每次处理后T变为原来的0.99，直到T<Tmin  模拟退火算法结束
    listx和listy是为了追踪互信息熵值随着迭代次数变化规律 而设置的两个列表
    '''
    T = 1000
    Tmin = 1e-4
    rate = 0.99
    count = 1
    listy = []
    listy.append(Z)
    # listx = [i for i in range(2728)]
    while T > Tmin:
        T *= rate  # 温度变化
        for i in range(5):  # 在此温度下迭代5次
            x = X + random.randint(int(-sigmodFun(T) * candidate_x),
                                   int(sigmodFun(T) * candidate_x))
            y = Y + random.randint(int(-sigmodFun(T) * candidate_y), int(sigmodFun(T) * candidate_y))
            if (x >= startX and x <= endX and y >= startY and y <= endY):  # 判断是否越界

                z = targetImg(distribution_template, x, y, target, template_x, template_y)
                if z > Z:  # 如果互信息值增大 则交换
                    X = x
                    Y = y
                    Z = z
                else:
                    if math.exp(-(Z - z) / (T * rate)) > random.random():  # 如果互信息值减小 以一定概率交换
                        X = x
                        Y = y
                        Z = z
                print(T, end=",")
                print(count, end=":")
                print(Z)
                listy.append(Z)
                count = count + 1

    # 下面这几行的作用就是：画出互信息值随着迭代次数的增加的变化规律
    listx = [i for i in range(count)]
    plt.plot(listx, listy)
    plt.grid(.5)
    plt.show()
    print(X, Y, Z)

    index = (Y, X)
    cv2.rectangle(target, index, (index[0] + template_y, index[1] + template_x), (0, 255, 255), 5)

    patt = "./img03/" + ss
    cv2.imwrite(patt, target)


'''
此函数的作用是：计算出在目标图像中坐标为i，j的点的子图像与模板图像的互信息值
'''


def targetImg(distribution_template, i, j, target, template_x, template_y):
    distribution_target = np.zeros(256, np.int32)
    for m in range(template_x):
        for n in range(template_y):
            # 统计第i行第j列的候选框的每个灰度值的数量
            distribution_target[target[i + m, j + n]] += 1

    # print('4.计算互信息熵值')
    IdS = metrics.mutual_info_score(distribution_template, distribution_target)
    return IdS


def sigmodFun(T):
    return 1 / (1 + math.e ** (-50 * (T - 0.02)))


def main():
    # 读取图片并进行处理
    path = "./img01"
    readImg(path)

    # 读取处理后的图片，并合成为视频
    path2 = "./img03"
    images_to_video(path2)


if __name__ == "__main__":
    main()
