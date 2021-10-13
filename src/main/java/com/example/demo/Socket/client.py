import json
from socket import *

import cv2 as cv
import numpy
import numpy as np
import tensorflow as tf
import time
from PIL import Image
from matplotlib import pyplot as plt, cm


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#
def printtime(str):
    ISOTIMEFORMAT = '[%Y-%m-%d %X]'
    print(time.strftime(ISOTIMEFORMAT, time.localtime()), str)


batch_size = 100  # 一个批次的大小，可优化


# 权值初始化
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))  # 截断的正态分布


# 偏置值初始化
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 卷基层
def conv2d(x, W):
    # x表示输入,是一个张量，W表示过滤器，即卷积核，strides表示步长，padding表示是否补零
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 获取图片像素矩阵
def getImageFromFile(filename):
    im = np.fromfile(filename, np.byte)
    im = im.reshape(1, 784)
    return im


# # 开启一个tcp监听，接收web应用发送过来的图片，进行数字预测，并返回给web应用
# def socket_server():
#     HOST = '127.0.0.1'
#     PORT = 50008
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind((HOST, PORT))
#     s.listen(1)
#     while 1:
#         print("wait for connect...")
#         conn, addr = s.accept()
#         print('Connected by', addr)
#         conn.sendall(predict())
#         conn.close()

#
def pre_pic(picName):
    # 将源图片转化为适合喂入神经网络的[1,784]格式
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    # 色反
    # threshold = 50
    # for i in range(28):
    #     for j in range(28):
    #         im_arr[i][j] = 255 - im_arr[i][j]
    #         if im_arr[i][j] < threshold:
    #             im_arr[i][j] = 0
    #         else:
    #             im_arr[i][j] = 255

    # 可视化
    # plt.imshow(im_arr)
    # plt.show()

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, (1.0 / 255.0))

    return img_ready


# def pre_pic(picName):
#     img = Image.open(picName)
#     reIm = img.resize((28,28), Image.ANTIALTAS)
#
#     im_arr = np.array(reIm.convert('L'))
#     threshold = 50   #设定合理的阈值
#     for i in range(28):
#         for j in range(28):
#             #255-原值使白底黑字变黑底白字
#             im_arr[i][j] = 255 - im_arr[i][j]
#             if(im_arr[i][j] < threshold):
#                 im_arr[i][j] = 0;
#             else:
#                 im_arr[i][j] = 255
#     nm_arr = im_arr.reshape([1, 784])
#     nm_arr = nm_arr.astype(np.float32)
#     img_ready = np.multiply(nm_arr, 1.0/255.0)
#
#     return img_ready

# 预测图片中的数字

def predict_number():
    predicted = []
    # 创建一个空的9x9数组
    sudoku = [[0 for i in range(9)] for j in range(9)]
    count = 0
    count2 = 0
    # 从约定的路径中读取待识别图片

    # img = pre_pic(
    #     'C:\\Users\\Administrator\\Desktop\\testimage\\optimized\\14.jpg')
    # image = getImageFromFile(img)

    # 加载图片到cnn神经网络计算，返回最大概率的分类，即为最大可能的数字
    # 依次读取文件夹内所有图片，预测完毕后写入数组
    for i in range(1, 82):
        img = pre_pic('C:\\Users\\Administrator\\Desktop\\testimage\\optimized\\' + str(i) + '.jpg')
        printtime("predicit:")
        prediction = tf.argmax(predict, 1)  # y_conv
        ret = sess.run([prediction, predict], feed_dict={x: img, keep_prob: 0.5})
        max = ret[0][0]
        predicted.append(max)
        print(max)
        count2 = count2 + 1
        # cv.imshow("test",img)
        # cv.waitKey(0)

    print(predicted)
    print(count2)

    # 将预测好的数字分配给9x9
    for line in range(9):
        for row in range(9):
            sudoku[line][row] = predicted[count]
            print(count)
            count = count + 1

    print(predicted)
    return sudoku


#######################################################################################################


# 启动tcp监听，识别web应用发送过来的图片
# socket_server()

# predict()


###########################################################################
def show_img(img):
    plt.imshow(img)
    plt.show()


def show_gray_img(img):
    plt.imshow(img, cmap=cm.gray)
    plt.show()


def count_number(num_list, num):
    """
    统计一维数组中某个数字的个数
    :param num_list:
    :param num:
    :return: num的数量
    """
    t = 0
    for i in num_list:
        if i == num:
            t += 1
    return t


def cut_level(img, cvalue=255, threshold=7):
    """
    投影法水平切割一张图片 主要处理多行文本
    :param cvalue:  切割线的颜色
    :param img: 传入为一张图片
     threshold 越大，字符间距越大，越大概
    :return: 水平切割之后的图片数组
    """
    r_list = []
    end = 0
    for i in range(len(img)):
        if count_number(img[i], cvalue) >= img.shape[1]:
            star = end
            end = i
            if end - star > threshold:
                # 如果相差值大于一的时候就说明跨过待切割的区域，
                # 根据 star 和end 的值就可以获取区域
                r_list.append(img[star:end, :])
    return r_list


def cut_vertical(img_list, cvalue=255, threshold=5):
    """
    投影法竖直切割图片的数组
    :param img_list: 传入的数据为一个由（二维）图片构成的数组，不是单纯的图片
    :param cvalue: 切割的值 同cut_level中的cvalue
    threshold 越大，字符间距越大，越大概
    :return: 切割之后的图片的数组
    """
    # 如果传入的是一个普通的二值化的图片，则需要首先将这个二值化的图片升维为图片的数组
    if len(np.array(img_list).shape) == 2:
        img_list = img_list[None]
    r_list = []
    for img_i in img_list:
        end = 0
        for i in range(len(img_i.T)):
            if count_number(img_i.T[i], cvalue) >= img_i.shape[0]:
                star = end
                end = i
                if end - star > threshold:
                    r_list.append(img_i[:, star:end])
    return r_list


def cut_image_by_projection(img, cvalue=255, patern=2):
    """
    传入二值化处理之后的图片 通过投影切割获取每个单独的数字
    处理方法默认为先水平切割再竖直切割
    :param cvalue: 根据切个数值，默认为255（根据白色切割），可选择0（根据黑色切割）
    :param img:传入的二值化图片
    :param patern: 2 为水平竖直两次切割，0 为水平切割， 1 为竖直切割
    :return: 返回切割完成后的图片数组
    """
    if patern == 2:
        return cut_vertical(cut_level(img, cvalue=cvalue), cvalue=cvalue)
    elif patern == 1:
        return cut_vertical(img, cvalue=cvalue)
    else:
        return cut_level(img, cvalue=cvalue)


def textureSquare(image):
    """
    将图片补充为正方形
    """
    # image = Image.open(path)
    image = image.convert('RGB')
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)),
                           color=(255, 255, 255))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)
    image_data = background.resize((28, 28))  # 缩放
    # image_data.show()
    return image_data


def unevenLightCompensate(img, blockSize):
    """
    光照不均匀处理
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv.GaussianBlur(dst, (3, 3), 0)
    dst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    return dst


def denoise(img):
    """
    去噪
    """
    kernel = np.ones((5, 5), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.blur(img, (10, 10))
    return img


def erode(img):
    """
    腐蚀（减小高亮部分）
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(img, kernel)
    return dst


def graying(img):
    """
    灰度化
    二值化
    :param img:
    :return:
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 1)
    img = thresh
    return thresh


def boost(img):
    """
    增强对比度 全局直方图均衡化
    :param img:
    :return:
    """
    img_equalize = cv.equalizeHist(img)
    # cv.imshow("img", img)
    # cv.imshow("img_equalize", img_equalize)
    # cv.waitKey(0)
    return img_equalize


def liner(img):
    """
   增强对比度 线性变换 y = ax + b
   :param img:
   :return:
   """
    dst = cv.convertScaleAbs(img, alpha=1.75, beta=0)

    return dst


def interference_point(img):
    """
    领域降噪
    :param img:
    :return:
    """
    # todo 判断图片的长宽度下限
    x = 0
    y = 0
    cur_pixel = img[x, y]  # 当前像素点的值
    height, width = img.shape[:2]
    for y in range(0, width - 1):
        for x in range(0, height - 1):
            if y == 0:  # 第一行
                if x == 0:  # 左上顶点,4邻域
                    # 中心点旁边3个点
                    sum = int(cur_pixel) \
                          + int(img[x, y + 1]) \
                          + int(img[x + 1, y]) \
                          + int(img[x + 1, y + 1])
                    if sum <= 2 * 245:
                        img[x, y] = 0
                elif x == height - 1:  # 右上顶点
                    sum = int(cur_pixel) \
                          + int(img[x, y + 1]) \
                          + int(img[x - 1, y]) \
                          + int(img[x - 1, y + 1])
                    if sum <= 2 * 245:
                        img[x, y] = 0
                else:  # 最上非顶点,6邻域
                    sum = int(img[x - 1, y]) \
                          + int(img[x - 1, y + 1]) \
                          + int(cur_pixel) \
                          + int(img[x, y + 1]) \
                          + int(img[x + 1, y]) \
                          + int(img[x + 1, y + 1])
                    if sum <= 3 * 245:
                        img[x, y] = 0
            elif y == width - 1:  # 最下面一行
                if x == 0:  # 左下顶点
                    # 中心点旁边3个点
                    sum = int(cur_pixel) \
                          + int(img[x + 1, y]) \
                          + int(img[x + 1, y - 1]) \
                          + int(img[x, y - 1])
                    if sum <= 2 * 245:
                        img[x, y] = 0
                elif x == height - 1:  # 右下顶点
                    sum = int(cur_pixel) \
                          + int(img[x, y - 1]) \
                          + int(img[x - 1, y]) \
                          + int(img[x - 1, y - 1])

                    if sum <= 2 * 245:
                        img[x, y] = 0
                else:  # 最下非顶点,6邻域
                    sum = int(cur_pixel) \
                          + int(img[x - 1, y]) \
                          + int(img[x + 1, y]) \
                          + int(img[x, y - 1]) \
                          + int(img[x - 1, y - 1]) \
                          + int(img[x + 1, y - 1])
                    if sum <= 3 * 245:
                        img[x, y] = 0
            else:  # y不在边界
                if x == 0:  # 左边非顶点
                    sum = int(img[x, y - 1]) \
                          + int(cur_pixel) \
                          + int(img[x, y + 1]) \
                          + int(img[x + 1, y - 1]) \
                          + int(img[x + 1, y]) \
                          + int(img[x + 1, y + 1])

                    if sum <= 3 * 245:
                        img[x, y] = 0
                elif x == height - 1:  # 右边非顶点
                    sum = int(img[x, y - 1]) \
                          + int(cur_pixel) \
                          + int(img[x, y + 1]) \
                          + int(img[x - 1, y - 1]) \
                          + int(img[x - 1, y]) \
                          + int(img[x - 1, y + 1])

                    if sum <= 3 * 245:
                        img[x, y] = 0
                else:  # 具备9领域条件的
                    sum = int(img[x - 1, y - 1]) \
                          + int(img[x - 1, y]) \
                          + int(img[x - 1, y + 1]) \
                          + int(img[x, y - 1]) \
                          + int(cur_pixel) \
                          + int(img[x, y + 1]) \
                          + int(img[x + 1, y - 1]) \
                          + int(img[x + 1, y]) \
                          + int(img[x + 1, y + 1])
                    if sum <= 4 * 245:
                        img[x, y] = 0
    return img


def cut(image_path):

    img = cv.imread(image_path)
    # 光照处理
    light = unevenLightCompensate(img, 5)
    # cv.imwrite(
    #     'C:\\Users\\Administrator\\Desktop\\testimage\\light1.jpg', light)
    # light = unevenLightCompensate(img, 10)
    # cv.imwrite(
    #     'C:\\Users\\Administrator\\Desktop\\testimage\\light2.jpg', light)
    # light = unevenLightCompensate(img, 15)
    # cv.imwrite(
    #     'C:\\Users\\Administrator\\Desktop\\testimage\\light3.jpg', light)
    # light = unevenLightCompensate(img, 17)
    # cv.imwrite(
    #     'C:\\Users\\Administrator\\Desktop\\testimage\\light4.jpg', light)
    img = unevenLightCompensate(img, 22)
    cv.imwrite(
        'C:\\Users\\Administrator\\Desktop\\testimage\\light5.jpg', img)
    # 增强对比度
    # img = boost(img)
    # 增强对比度
    img = liner(img)
    # 灰度化 二值化
    img = graying(img)
    # 均衡化
    img = cv.equalizeHist(img)
    # 平滑
    # img = cv.GaussianBlur(img,(3,3),1)
    # 去噪(模糊)
    # img = denoise(img)
    # 腐蚀
    # img = erode(img)
    # 锐化
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 155, 1)

    # 八领域去噪点
    img = interference_point(img)

    img3 = denoise(img)
    # 1.均值滤波
    blur = cv.blur(img, (3, 3))
    # 2.高斯滤波
    gau_blur = cv.GaussianBlur(img, (3, 3), 0)

    # cv.imshow("0", img)
    # cv.imshow("1", img1)
    # cv.imshow("2", img2)
    # cv.imshow("3", img3)
    # cv.imshow("4", blur)
    # cv.imshow("5", gau_blur)
    # cv.imshow("5", img_gauss)
    # cv.waitKey(0)

    cv.imwrite(
        'C:\\Users\\Administrator\\Desktop\\testimage\\prprprpr.jpg', img)
    img_list = cut_image_by_projection(img)
    t = 1

    for i in img_list:
        # CV图片转PIL
        PIL_img = Image.fromarray(cv.cvtColor(i, cv.COLOR_BGR2RGB))
        # 正方化
        square_img = textureSquare(PIL_img)

        # pil转cv
        cv_img = cv.cvtColor(np.asarray(square_img), cv.COLOR_RGB2BGR)
        # 色反操作
        cv_img = cv.bitwise_not(cv_img)

        # 这里可以对切割到的图片进行操作，显示出来或者保存下来
        cv.imwrite(
            'C:\\Users\\Administrator\\Desktop\\testimage\\optimized\\' + str(t) + '.jpg', cv_img)
        print("successful cut " + str(t) )
        t += 1
    return "succeed cutting"


def socket_(sudoku):
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSocket.bind(("", 6000))
    serverSocket.listen(1)
    print('Waiting for connection...')
    tcpCliSock, addr = serverSocket.accept()

    print('...connected from', addr)
    message = tcpCliSock.recv(2048)
    print("receive : ", message)
    # list = [[1, 3, 4, 5], [4, 5, 6, 7], [1, 5, 6]]
    list = sudoku
    json_string = json.dumps(list, cls=MyEncoder)
    # a = cut()
    str = json.dumps(json_string) + '\0'
    print(str)
    tcpCliSock.send(str.encode())
    tcpCliSock.close()


def main(sudoku):
    """
    服务端，发送预测完成的数独
    :param sudoku:
    :return:
    """
    client = socket()  # 声明socket类型，同时生成socke连接t对象
    client.connect(('localhost', 6002))  # 连接到localhost主机的6969端口上去
    # while True:
    #     msg = input(">>:").strip()
    #     if len(msg) == 0:continue
    #     server.send(msg.encode('utf-8'))#把编译成utf-8的数据发送出去
    #     data = server.recv(512)#接收数据
    #     print("从服务器接收到的数据为：",data.decode())


    # serverSocket = socket(AF_INET, SOCK_STREAM)
    # serverSocket.setsockopt(SOL_SOCKET,SO_REUSEADDR,1)


    # server.connect(('localhost',6969))
    # serverSocket.bind(("",6002))
    # serverSocket.listen(1)
    # print ('Waiting for connection...')
    # tcpCliSock,addr = serverSocket.accept()
    #
    # print('...connected from',addr)
    # message = tcpCliSock.recv(1024)
    # print("receive : ", message)

    # list = [[1, 3, 4, 5], [4, 5, 6, 7], [1, 5, 6]]
    list = sudoku
    json_string = json.dumps(list, cls=MyEncoder)
    # a = cut()
    str = json.dumps(json_string) + '\0'
    print(str)

    client.send(str.encode())
    # client.close()


if __name__ == '__main__':
    # 初始化tensorflow的session
    sess = tf.InteractiveSession()

    # # 按照LeNet-5模型定义卷积神经网络的各层
    # x = tf.placeholder("float", shape=[None, 784])
    # y_ = tf.placeholder("float", shape=[None, 10])
    #
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    #
    # # 卷积层
    # W_conv1 = weight_variable([5, 5, 1, 32])
    # b_conv1 = bias_variable([32])
    #
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    #
    # # 池化层
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    #
    # # 卷积层
    # W_conv2 = weight_variable([5, 5, 32, 64])
    # b_conv2 = bias_variable([64])
    #
    # # 池化层
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    #
    # W_fc1 = weight_variable([7 * 7 * 64, 1024])
    # b_fc1 = bias_variable([1024])
    #
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    # keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    # # 全连接softmax层
    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 定义两个占位符（训练数据集和标签）
    x = tf.placeholder(tf.float32, [None, 784])  # 28*28
    y = tf.placeholder(tf.float32, [None, 10])

    # 将x转化为4d向量[batch,height,width,channel]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 初始化第一个卷积层的权值(卷积核)和偏置
    w_con1 = weight_variable([5, 5, 1, 32])  # 5*5窗口大小，从一个平面中进行32次卷积，抽取32次特征
    b_con1 = bias_variable([32])  # 每一次卷积对应一个偏置，所以32

    # 将输入和权值进行卷积，加上偏置后激活，再进行池化
    h_con1 = tf.nn.relu(conv2d(x_image, w_con1) + b_con1)
    h_pool1 = max_pool_2x2(h_con1)

    # 初始化第二个卷积层的权值(卷积核)和偏置
    w_con2 = weight_variable([5, 5, 32, 64])  # 5*5窗口大小，64次卷积从32个平面中抽取64次特征
    b_con2 = bias_variable([64])  # 每一次卷积对应一个偏置，所以32

    # 将输入和权值进行卷积，加上偏置后激活，再进行池化
    h_con2 = tf.nn.relu(conv2d(h_pool1, w_con2) + b_con2)
    h_pool2 = max_pool_2x2(h_con2)

    # 28*28*1的图片第一次卷积后tensor的shape是28*28*32，第一次池化后是14*14*32
    # 第二次卷积后输出shape是14*14*64，第二次池化后是7*7*64

    # 第一个全连接层的权值和偏置
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # 池化后的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # 第一个全连接层的输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # 神经元输出概率，防止过拟合
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 第二个全连接层
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    predict = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)  # softmax将输出信号转化为概率值（10个概率值）

    # 定义tensorflow的计算
    init = tf.global_variables_initializer()
    sess.run(init)

    # 加载模型参数文件
    saver = tf.train.Saver()
    saver.restore(sess, "D:/作业/练习/python/model/mnist")

    # D:/作业/练习/cnn_number/mnist_variables/variables.ckpt

    cut('C:\\Users\\Administrator\\Desktop\\testimage\\uploaded\\image.jpg')
    # cut()
    # cut('C:\\Users\\Administrator\\Desktop\\testimage\\image9.jpg')
    # print(predict_number())
    # socket_(predict_number())

    main(predict_number())

