import cv2
import matplotlib.pyplot as plt
import os
import numpy as np 
# Q5. Find the external boundary of coins from “coins.jpg” image. (a) At first, you should perform the edge detection. (b) Then, find the external boundary of coins. Report detail of each step to final output

# Read a image
if os.path.isdir('Assignment5') == False :
    os.makedirs('Assignment5')

# Read a Image
coins = cv2.imread('Images\\coins.jpg', 0)
coins_left = coins[:, :188]
coins_right = coins[:, 198: ]

# Make a kernel
sobelkernel = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3)
laplaciankerenl = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape(3, 3)
gaussiankernel = cv2.getGaussianKernel(ksize=9, sigma=3).reshape(3, 3)

# Extract a edge
coins_left_gaussian =cv2.filter2D(coins_left, -1, gaussiankernel)
coins_left_sub = coins_left - coins_left_gaussian
coins_left_sobel= cv2.filter2D(coins_left, -1, sobelkernel)
coins_left_laplacian = cv2.filter2D(coins_left, -1, laplaciankerenl)
coins_left_canny = cv2.Canny(coins_left, 0, 200)

# Concat a target image and maked image 
coins_concat = np.concatenate((coins_right, coins_left_canny), axis=1)

# Make a edge detected image
cv2.imwrite('Assignment5\\coins_left.png',coins_left)
cv2.imwrite('Assignment5\\coins_right.png',coins_right)
cv2.imwrite('Assignment5\\coins_left_sub.png',1.5 * coins_left_sub)
cv2.imwrite('Assignment5\\coins_left_sobel.png',coins_left_sobel)
cv2.imwrite('Assignment5\\coins_left_laplacian.png',coins_left_laplacian)
cv2.imwrite('Assignment5\\coins_left_canny.png',coins_left_canny)
cv2.imwrite('Assignment5\\coins_concat.png',coins_concat)
