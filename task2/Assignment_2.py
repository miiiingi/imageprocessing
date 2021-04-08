import cv2
import matplotlib.pyplot as plt
import os
import numpy as np 
# Q2. Remove the noise from four images (a) “cb1.tif”, (b) “cb2.tif”, (c) “cb3.tif”, (d) “cb4.tif”. What was the most effective method to remove noise from each image? Compare the results and explain it. 

# Read a image
if os.path.isdir('Assignment2') == False :
    os.makedirs('Assignment2')

cb1 = cv2.imread('Images\\cb1.tif')
cb2 = cv2.imread('Images\\cb2.tif')
cb3 = cv2.imread('Images\\cb3.tif')
cb4 = cv2.imread('Images\\cb4.tif')

cv2.imwrite('Assignment2\\cb1.png',cb1)
cv2.imwrite('Assignment2\\cb2.png',cb2)
cv2.imwrite('Assignment2\\cb3.png',cb3)
cv2.imwrite('Assignment2\\cb4.png',cb4)

# To Check a distribution of intensity
plt.clf()
hist_cb1 = cv2.calcHist([cb1], [0], None, [256], [0, 256])
plt.plot(hist_cb1)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb1.png')

# Make a Blured Image
cb1_G = cv2.GaussianBlur(cb1, (3,3), 0)
cb1_M = cv2.medianBlur(cb1, 3)
cb1_M5 = cv2.medianBlur(cb1, 5)
cb1_MM = cv2.medianBlur(cb1_M, 3)

cb1_concat1 = np.concatenate((cb1, cb1_G, cb1_M5), axis=1)
cb1_concat2 = np.concatenate((cb1_M, cb1_MM, cb1_M5), axis=1)
cb1_result = np.concatenate((cb1, cb1_MM), axis=1)

cv2.imwrite('Assignment2\\cb1_concat_OGM.png', cb1_concat1)
cv2.imwrite('Assignment2\\cb1_concat_M.png', cb1_concat2)
cv2.imwrite('Assignment2\\cb1_result.png', cb1_result)

# Graph a intensity histogram(after blurring)
plt.clf()
hist_cb1_result = cv2.calcHist([cb1_result], [0], None, [256], [0, 256])
plt.plot(hist_cb1_result)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb1_result.png')

# To Check a distribution of intensity
plt.clf()
hist_cb2 = cv2.calcHist([cb2], [0], None, [256], [0, 256])
plt.plot(hist_cb2)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb2.png')

# Make a Blured Image
cb2_G = cv2.GaussianBlur(cb2, (5,5), 0)
cb2_M = cv2.medianBlur(cb2, 3)
cb2_M5 = cv2.medianBlur(cb2, 5)
cb2_MM = cv2.medianBlur(cb2_M, 3)
cb2_concat1 = np.concatenate((cb2, cb2_G, cb2_M5), axis=1)
cb2_concat2 = np.concatenate((cb2_M, cb2_MM, cb2_M5), axis=1)
cb2_result = np.concatenate((cb2, cb2_MM), axis=1)

cv2.imwrite('Assignment2\\cb2_concat_OGM.png', cb2_concat1)
cv2.imwrite('Assignment2\\cb2_concat_M.png', cb2_concat2)
cv2.imwrite('Assignment2\\cb2_result.png', cb2_result)

# Graph a intensity histogram(after blurring)
plt.clf()
hist_cb2_result = cv2.calcHist([cb2_result], [0], None, [256], [0, 256])
plt.plot(hist_cb2_result)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb2_result.png')

# To Check a distribution of intensity
plt.clf()
hist_cb3 = cv2.calcHist([cb3], [0], None, [256], [0, 256])
plt.plot(hist_cb3)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb3.png')

# Make a Blured Image
cb3_G = cv2.GaussianBlur(cb3, (5,5), 0)
cb3_M = cv2.medianBlur(cb3, 3)
cb3_M5 = cv2.medianBlur(cb3, 5)
cb3_MM = cv2.medianBlur(cb3_M, 3)
cb3_concat1 = np.concatenate((cb3, cb3_G, cb3_M5), axis=1)
cb3_concat2 = np.concatenate((cb3_M, cb3_MM, cb3_M5), axis=1)
cb3_result = np.concatenate((cb3, cb3_MM), axis=1)

cv2.imwrite('Assignment2\\cb3_concat_OGM.png', cb3_concat1)
cv2.imwrite('Assignment2\\cb3_concat_M.png', cb3_concat2)
cv2.imwrite('Assignment2\\cb3_result.png', cb3_result)

# Graph a intensity histogram(after blurring)
plt.clf()
hist_cb3_result = cv2.calcHist([cb3_result], [0], None, [256], [0, 256])
plt.plot(hist_cb3_result)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb3_result.png')

# To Check a distribution of intensity
plt.clf()
hist_cb4 = cv2.calcHist([cb4], [0], None, [256], [0, 256])
plt.plot(hist_cb4)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb4.png')

# Make a Blured Image
cb4_G = cv2.GaussianBlur(cb4, (5,5), 0)
cb4_M = cv2.medianBlur(cb4, 3)
cb4_M5 = cv2.medianBlur(cb4, 5)
cb4_MM = cv2.medianBlur(cb4_M, 3)
cb4_concat1 = np.concatenate((cb4, cb4_G, cb4_M5), axis=1)
cb4_concat2 = np.concatenate((cb4_M, cb4_MM, cb4_M5), axis=1)
cb4_result = np.concatenate((cb4, cb4_MM), axis=1)

cv2.imwrite('Assignment2\\cb4_concat_OGM.png', cb4_concat1)
cv2.imwrite('Assignment2\\cb4_concat_M.png', cb4_concat2)
cv2.imwrite('Assignment2\\cb4_result.png', cb4_result)

# Graph a intensity histogram(after blurring)
plt.clf()
hist_cb4_result = cv2.calcHist([cb4_result], [0], None, [256], [0, 256])
plt.plot(hist_cb4_result)
plt.xlim([0,256])
plt.savefig('Assignment2\\hist_cb4_result.png')
