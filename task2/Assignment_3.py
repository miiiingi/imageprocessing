import cv2
import matplotlib.pyplot as plt
import os
import numpy as np 
# Q3. Conduct a series of experiments to determine how the sigma value affects the performance of gaussian filter, then develop an idea of how you might automatically determine the parameter depending on image content. 

# Make a directory
if os.path.isdir('Assignment3') == False :
    os.makedirs('Assignment3')

# Read a image
Lenna = cv2.imread('Images\\Lenna.png')
saltpepper = cv2.imread('Images\\cb4.tif')

# Define the various kernels
kernel_1D_1 = cv2.getGaussianKernel(ksize=27, sigma=1)
kernel_2D_1 = np.outer(kernel_1D_1, kernel_1D_1.transpose())
kernel_1D_3 = cv2.getGaussianKernel(ksize=27, sigma=3)
kernel_2D_3 = np.outer(kernel_1D_3, kernel_1D_3.transpose())
kernel_1D_9 = cv2.getGaussianKernel(ksize=27, sigma=9)
kernel_2D_9 = np.outer(kernel_1D_9, kernel_1D_9.transpose())
kernel_1D_27 = cv2.getGaussianKernel(ksize=27, sigma=27)
kernel_2D_27 = np.outer(kernel_1D_27, kernel_1D_27.transpose())

# Implement to filter the images
Lenna_1= cv2.filter2D(Lenna, -1, kernel_2D_1)
Lenna_3= cv2.filter2D(Lenna, -1, kernel_2D_3)
Lenna_9= cv2.filter2D(Lenna, -1, kernel_2D_9)
Lenna_27= cv2.filter2D(Lenna, -1, kernel_2D_27)

saltpepper_1= cv2.filter2D(saltpepper, -1, kernel_2D_1)
saltpepper_3= cv2.filter2D(saltpepper, -1, kernel_2D_3)
saltpepper_9= cv2.filter2D(saltpepper, -1, kernel_2D_9)
saltpepper_27= cv2.filter2D(saltpepper, -1, kernel_2D_27)

# concat the generated images
Lenna_concat = np.concatenate((Lenna, Lenna_1, Lenna_3, Lenna_9, Lenna_27), axis=1)
saltpepper_concat = np.concatenate((saltpepper, saltpepper_1, saltpepper_3, saltpepper_9, saltpepper_27), axis=1)
cv2.imwrite('Assignment3\\Lenna_concat.png',Lenna_concat)
cv2.imwrite('Assignment3\\saltpepper_concat.png',saltpepper_concat)