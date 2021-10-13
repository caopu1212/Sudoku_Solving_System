'''
Author: your name
Date: 2020-06-30 11:39:31
LastEditTime: 2020-08-10 17:17:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \练习\python\demo.py
'''
import cv2
import numpy as np


def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    blockImage2 = cv2.resize(
        blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst


def erode(image):
    """
    腐蚀（减小高亮部分）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(
        gray, 135, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.erode(binary, kernel)
    cv2.imshow("erode", dst)
    return dst

if __name__ == '__main__':
    blockSize = 16
    img = cv2.imread(
        'C:\\Users\\Administrator\\Desktop\\testimage\\demoo01.jpg')
    dst = unevenLightCompensate(img, blockSize)
    result = np.concatenate([img, dst], axis=1)



    cv2.imwrite(
        'C:\\Users\\Administrator\\Desktop\\testimage\\demooo3.jpg',erode(dst))

