import matplotlib.pyplot as plt
import cv2


imgBGR = cv2.imread('amongus.png')
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)        # BGR to RGB
imgGray = cv2.imread('amongus.png', cv2.IMREAD_GRAYSCALE)


plt.axis('off')         # x축, y축 눈금 비활성화
# plt.imshow(imgRGB)
# plt.imshow(imgBGR)
# plt.show()

plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB)
plt.subplot(122), plt.axis('off'), plt.imshow(imgGray, cmap='gray')
plt.show()