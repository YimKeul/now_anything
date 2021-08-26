import sys
import cv2
import numpy as np
import matplotlib.pylab as plt

src = cv2.imread('img\lion.jpg', cv2.IMREAD_COLOR)

image = cv2.imread('img\lion.jpg')

# mask = cv2.imread('img\lion.jpg', cv2.IMREAD_GRAYSCALE)
mask = np.zeros(image.shape[:2], dtype="uint8")
dst = cv2.imread('img\img1.jpg' , cv2.IMREAD_COLOR)


cv2.namedWindow('src', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)

cv2.resizeWindow('src', 500,500)
cv2.resizeWindow('mask', 500,500)
cv2.resizeWindow('dst', 500,500)

cv2.copyTo(src, mask, dst)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)


cv2.waitKey()
cv2.destroyAllWindows()