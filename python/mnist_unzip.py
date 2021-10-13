#!/usr/bin/env python 3.6
#_*_coding:utf-8 _*_
#@Time    :2019/12/30 16:17
#@Author  :控制工程小小白
#@FileName: class.py

#@Software: PyCharm
import torchvision
from skimage import io
#import os
mnist_train=torchvision.datasets.MNIST('./make_mnistdata',train=True,download=True)#首先下载数据集，并数据分割成训练集与数据集
mnist_test=torchvision.datasets.MNIST('./make_mnistdata',train=False,download=True)
#print('testset:',len(mnist_test))
#txt_path = "F:\桌面文件\make_Mnist_data"
# if not os.path.exists(txt_path):
#     os.makedirs(txt_path)
f=open("./mnist_train.txt",'w')#在指定路径之下生成.txt文件
"""这个是对图片与标签分别保存"""
"""for i,(img,label) in enumerate(mnist_train):
    img_path = "./mnist_train/" + str(i) + ".jpg"
    io.imsave(img_path, img)#将图片数据以图片.jpg格式存在指定路径下
    img_paths=str(i)+".jpg"
    f.write(str(label)+'，')#将路径与标签组合成的字符串存在.txt文件下
f.close()#关闭文件"""
"""这个是对相同的数据保存在同一个文件夹下"""
for i,(img,label) in enumerate(mnist_train):
    img_path = r"F:/桌面文件/make_Mnist_data"+"/"+str(label)+"/" + str(i) + ".jpg"
    io.imsave(img_path, img)#将图片数据以图片.jpg格式存在指定路径下
    img_paths=str(i)+".jpg"
    f.write(str(label)+'，')#将路径与标签组合成的字符串存在.txt文件下
f.close()#关闭文件