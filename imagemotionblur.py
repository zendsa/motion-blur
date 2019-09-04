# motion blur
from scipy import mgrid,exp
import numpy as np
from numpy.fft import *
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("E:/python/my programs/cnn/imagemodel/Test/character_1_ka/10963.png",0)

epsilon = 0.00001
r=1000

sizeY,sizeX = img.shape
blur = cv2.GaussianBlur(img,(5,5),0)

"""def makeGaussianPSF(radius,X,Y):
     Returns a normalized 2D gauss kernel array for convolutions 
    x,y = mgrid[-sizeY/2:sizeY/2, -sizeX/2:sizeX/2]
    g = exp(-(x**2/float(radius)+y**2/float(radius)))
    return(g / g.sum())  
psf = makeGaussianPSF(r,sizeX,sizeY)

kernel = np.ones((5,5),np.float32)/25
dst1 = cv2.filter2D(blur,-1,kernel)"""



# function of motion blur
def makeMotionPSF(length,X,Y):
    psf = np.zeros((sizeY,sizeX))
    psf[int(sizeY/2):int(sizeY/2+1),int(sizeX/2-length/2):int(sizeX/2+length/2)] = 1
    return(psf/psf.sum())
result = makeMotionPSF(20,sizeX,sizeY)
kernel = result
dst2 = cv2.filter2D(blur,-1,kernel)

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst2, cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()

