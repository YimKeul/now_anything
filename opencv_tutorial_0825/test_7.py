import sys
import cv2


# 마스크 영상을 이용한 영상 합성
src = cv2.imread('lion2.jpg' , cv2.IMREAD_COLOR)
mask = cv2.imread('mask_lion2.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('nature.jpg', cv2.IMREAD_COLOR)

cv2.copyTo(src,mask,dst)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)


cv2.waitKey()
cv2.destroyAllWindows()