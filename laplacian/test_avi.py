import numpy as np
import cv2

from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        continue

    frame = cv2.resize(frame, (320,240))
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    #모폴로지 (블러효과)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(gray.copy(), kernel)

    #캐니
    edges = cv2.Canny(gray.copy(),threshold1=100, threshold2=200,apertureSize = 3 , L2gradient=True )

    
    sobelxy = cv2.Sobel(gray.copy(),cv2.CV_8U, dx=1 , dy=1)
    sobelxy_1= cv2.Sobel(dilate,cv2.CV_8U, dx=1 , dy=1)

    #sobel
    laplacian_1 = cv2.Laplacian(dilate,cv2.CV_8U,13)
    
    laplacian = cv2.Laplacian(gray.copy(),cv2.CV_8U,13)
    # laplacian = cv2.Laplacian(gray.copy(),cv2.CV_64F )

    #kernel = np.ones((3,3),np.uint8)
    #laplacian = cv2.dilate(laplacian,kernel,iterations = 1)

    im_th1 = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 5, 2)
    
    # Display the resulting frame
    #edges = cv2.Canny(gray.copy(),threshold1=50, threshold2=150,apertureSize = 3)
    tempImg = cv2.medianBlur(gray, 5)
    im_th1 = cv2.adaptiveThreshold(tempImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 5, 2)

    blur = cv2.GaussianBlur(im_th1, (3,3), 0)
    _, im_th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
 
    numpy_horizontal = np.hstack((gray, edges, laplacian,laplacian_1, sobelxy,sobelxy_1 ))

    cv2.imshow('Numpy Horizontal', numpy_horizontal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()