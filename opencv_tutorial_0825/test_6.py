import sys
import cv2
import numpy as np
import matplotlib.pylab as plt
image = cv2.imread('lion2.jpg')

# src = cv2.imread('lion2.jpg', cv2.IMREAD_COLOR)
# dst = cv2.imread('nature.jpg' , cv2.IMREAD_COLOR)
# img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _,b_mask = cv2.threshold(img2gray,10,255, cv2.THRESH_BINARY)

# mask = cv2.bitwise_not(b_mask)


src = cv2.imread('lion2.jpg' , cv2.IMREAD_COLOR)
mask = cv2.imread('mask2_lion2.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('nature.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow('src', cv2.WINDOW_NORMAL)
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.resizeWindow('src', 500,500)
cv2.resizeWindow('dst', 500,500)
cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask',500,500)



cv2.copyTo(src,mask,dst)


cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)





cv2.waitKey()
cv2.destroyAllWindows()