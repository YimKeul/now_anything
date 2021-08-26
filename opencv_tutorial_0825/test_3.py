import sys
import cv2
import glob

img_files = glob.glob('.\\img\\*.jpg')

if not img_files:
  print("jpg 이미지가 없습니다")
  sys.exit()

cv2.namedWindow('nature',cv2.WINDOW_NORMAL)
cv2.setWindowProperty('nature', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



count = len(img_files)
index = 0

while True:
  img = cv2.imread(img_files[index])

  if img is None:
    print("이미지를 불러오는데 실패했습니다")
    break

  cv2.imshow('nature', img)
  if cv2.waitKey(1000) == 27:
    break

  index += 1
  if index >= count :
    index = 0
  
cv2.destroyAllWindows()