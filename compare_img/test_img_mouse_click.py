import sys
import numpy as np
import cv2
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
np.seterr(over='ignore')
# app = QtWidgets.QApplication([])
# label = QtWidgets.QLabel()

def average(o_t,o_r,o_g,o_b,c_t,c_r,c_g,c_b):


  pass

global c 
c = 1
def on_mouse(event, x, y,  flags, param):
  global c
  test1=cv2.cvtColor(img_o,cv2.COLOR_BGR2RGB)
  test2 = cv2.cvtColor(img_c,cv2.COLOR_BGR2RGB)
  num = c
  if event == cv2.EVENT_LBUTTONDOWN:
    #왼쪽사진을 클릭 기준으로 잡았음 
    # cv2.circle(addh, (x, y), 5, (0, 200, 255), -1)
    # cv2.circle(addh, (x+640, y), 5, (0, 0, 255), -1)


    #화면에 숫자 찍기
    cv2.putText(addh,str(num), (x, y), cv2.FONT_HERSHEY_SIMPLEX,  0.5 ,(255,255,255) , thickness=2)
    cv2.putText(addh,str(num), (x+640, y), cv2.FONT_HERSHEY_SIMPLEX,  0.5 ,(255,255,255) , thickness=2)
  
    
    cv2.imshow('image',addh)
    print("{}번 : 왼쪽 사진 : {}, {} 색상정보 : {} || 오른쪽 사진 : {}, {} 색상정보 : {}" .format(c,x,y,test1[y,x], x,y,test2[y,x]) )
    print("avg => r : {0:05.2f}% , g : {0:05.2f}% , b : {0:05.2f}% || total = {0:05.2f}%".format( (test1[y,x][0] - test2[y,x][0])/255 * 100 , (test1[y,x][1] - test2[y,x][1])/255 * 100 , (test1[y,x][2] - test2[y,x][2])/255 * 100 , 
    (  (test1[y,x][0] - test2[y,x][0])/255 * 100 + (test1[y,x][1] - test2[y,x][1])/255 * 100 + (test1[y,x][2] - test2[y,x][2])/255 * 100)/3   ) ) 
    
    
    c = c+1



img1 = cv2.imread('images/origin.png') #인감
img2 = cv2.imread('images/com.png') #어몽

img_o = cv2.resize(img1, dsize=(640,480)  ,interpolation=cv2.INTER_LINEAR) #인감
img_c = cv2.resize(img2, dsize=(640,480) , interpolation=cv2.INTER_LINEAR)   #어몽

img_o_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img_c_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


s = ssim(img_o_g, img_c_g)
print("SSIM 결과 : %.2f"%s)

#인감, 어몽 x
addh = cv2.hconcat([img_o,img_c])


cv2.imshow('image',addh)
cv2.setMouseCallback('image',on_mouse)


cv2.waitKey()
cv2.destroyAllWindows()
