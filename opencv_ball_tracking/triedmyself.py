import cv2

img = cv2.imread('C:/Users/mentp/vscode/vscode-now-workspace/opencv_ball_tracking/catchingball.png')
cv2.imshow("img", img)
k = cv2.waitKey(0)

#종료
if k == 27:
  cv2.destroyAllWindows()






