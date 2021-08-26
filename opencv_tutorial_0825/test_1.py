
import sys
import cv2
print("Hello Opencv", cv2.__version__)

img  = cv2.imread('amongus.png')

if img is None:
  print("이미지가 없습니다")
  sys.exit()

# cv2.imwrite('gray_amongus.png',img)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',10,10)
cv2.imshow('image',img)
cv2.waitKey()

cv2.destroyAllWindows()