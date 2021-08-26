import sys
import cv2

img1 = cv2.imread('amongus.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('amongus.png', cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
  print("이미지를 불러오는데 실패했습니다")
  sys.exit()

print(type(img1))
print(img1.shape)
print(img2.shape)
print(img1.dtype)
print(img2.dtype)

if len(img1.shape) == 2 :
    print("img1은 그레이스케일 이미지 입니다.") 
elif len(img1.shape) == 3 :
    print("img1는 컬러 이미지 입니다.")

# 영상의 크기 참조
h, w = img1.shape
print('img1 shape = w x h = {} x {}'.format(w, h))
h, w = img2.shape[:2]
#:2 하는 이유는 컬러 이미지는 차원수가 적혀있기 때문
print('img2 shape = w x h = {} x {}'.format(w, h))