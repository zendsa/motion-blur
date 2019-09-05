import numpy as np
import cv2
from matplotlib import pyplot as plt 

a = cv2.imread("E:/python/my programs/images/coded/ajayfont_a_rot/image1.jpg",0)

def get_mag_ang(img):

    img = np.sqrt(img)
    gx = cv2.Sobel(np.float32(img), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(img), cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)   
    return (mag, ang, gx, gy);

get_mag_ang(a)
plt.subplot(121),plt.imshow(a,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(a,cmap = 'gray')
plt.title('resultant Image'), plt.xticks([]), plt.yticks([])

plt.show()








