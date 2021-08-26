import numpy as np
import cv2

img1 = np.empty((240, 320) , dtype = np.uint8)    #그레이 스케일
img2 = np.zeros((240, 320, 3), dtype=np.uint8)    # 트루 컬러
img3 = np.ones((240, 320, 3), dtype=np.uint8) *255    # 트루 컬러
img4 = np.full((240, 320), 128, dtype=np.uint8)   # 그레이 스케일

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)

png1 = cv2.imread('C:/Users/mentp/vscode/vscode-now-workspace/opencv_tutorial_0825/img/img1.jpg')
png2 = png1[40:120, 30:150]
png3 = png1[40:120, 30:150].copy()

cv2.imshow('png1', png1)
cv2.imshow('png2', png2)
cv2.imshow('png3', png3)

cv2.waitKey()
cv2.destroyAllWindows()