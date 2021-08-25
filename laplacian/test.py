import numpy as np
import cv2



print(cv2.__version__)
path = 'C:/Users/mentp/vscode/vscode-AI-workspace/laplacian/lena.jpg'
# image = cv2.imread(path, cv2.IMREAD_UNCHAGED)

# cv2.imshow("image",image)
# cv2.waitKey()
# cv2.destoryAllWindows()

img = cv2.imread(path,0)

cv2.imshow('image',img)
cv2.waitKey(0)