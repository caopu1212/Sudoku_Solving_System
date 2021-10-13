'''
Author: your name
Date: 2020-08-25 16:00:34
LastEditTime: 2020-08-25 16:12:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \练习\python\straight_line.py
'''

import cv2
import numpy as np




img = cv2.imread("C:\\Users\\Administrator\\Desktop\\testimage\\image4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 0.5, np.pi / 180, 118)
result = img.copy()
cv2.imwrite("straight_line1.jpg",result)
# 经验参数
minLineLength = 200
maxLineGap = 15
lines = cv2.HoughLinesP(edges, 0.5, np.pi / 180, 80, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Origin", img)
cv2.imshow('Result', cv2.imread("straight_line1.jpg"))

cv2.waitKey(0)
cv2.destroyAllWindows()