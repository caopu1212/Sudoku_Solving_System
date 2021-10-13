'''
Author: your name
Date: 2020-08-25 16:04:11
LastEditTime: 2020-08-25 16:06:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \练习\python\smooth.py
'''

import cv2

img = cv2.imread("C:\\Users\\Administrator\\Desktop\\testimage\\image.jpg")


"""
低通滤波平滑
"""
result = cv2.boxFilter(img,-1, (5, 5))
cv2.imwrite("smoothing.jpg",result)

cv2.imshow("Origin", img)
cv2.imshow("Blur", cv2.imread("smoothing.jpg"))

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
高斯平滑
"""
result = cv2.GaussianBlur(img, (5, 5),1.5)
#元组为模糊范围，1.5为模糊度
cv2.imwrite("smoothing1.jpg",result)

cv2.imshow("Origin", img)
cv2.imshow("Blur", cv2.imread("smoothing1.jpg"))

cv2.waitKey(0)
cv2.destroyAllWindows()

"""
中值滤波平滑
"""
result = cv2.medianBlur(img,3)
cv2.imwrite("smoothing2.jpg",result)

cv2.imshow("Origin", img)
cv2.imshow("Blur", cv2.imread("smoothing2.jpg"))

cv2.waitKey(0)
cv2.destroyAllWindows()