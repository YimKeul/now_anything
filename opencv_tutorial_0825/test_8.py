import sys
import cv2
import numpy as np

image = cv2.imread('lion2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
_,threshold = cv2.threshold(gray, 200,255, cv2.THRESH_BINARY_INV)
# threswithblur = cv2.medianBlur(threshold,15,0)



cv2.namedWindow('change', cv2.WINDOW_NORMAL)
cv2.resizeWindow('change', 500,500)

cv2.namedWindow('change2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('change2', 500,500)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 500,500)

cv2.imshow('image',image)
cv2.imshow('change', gray)
cv2.imshow('change2', threshold)

cv2.imwrite('mask2_lion2.jpg', threshold)
cv2.waitKey()

cv2.destroyAllWindows()