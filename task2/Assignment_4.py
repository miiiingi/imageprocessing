import cv2
import matplotlib.pyplot as plt
import os
import numpy as np 
# Q4. The gray-scaled image “Berkeley.jpg” was taken in the shadow of a building. The camera couldn’t compensate for the bright blue sky and the dark shadow simultaneously. As a result, the sunlight areas are a glaring white, and the shadow areas are dark and low contrast. Using the image enhancement techniques, increase the contrast and the brightness of the dark areas, and reduce the brightness of the sunny areas

# Make a directory
if os.path.isdir('Assignment4') == False :
    os.makedirs('Assignment4')

# Read a image
Berkeley = cv2.imread('Images\\Berkeley.jpg', 0)

# Implement a histogram equalizing
Berkeley_equalizing = cv2.equalizeHist(Berkeley)
Berkeley_Aequalizing = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
Berkeley_Aequalizing = Berkeley_Aequalizing.apply(Berkeley)

# Concat a generated images
Berkeley_concat = np.concatenate((Berkeley, Berkeley_equalizing, Berkeley_Aequalizing), axis=1)
cv2.imwrite('Assignment4\\Berkeyley_concat.png', Berkeley_concat)

# Graph a intensity histogram
hist_Berkeley = cv2.calcHist([Berkeley], [0], None, [256], [0, 256])
hist_Berkeley_equalizing = cv2.calcHist([Berkeley_equalizing], [0], None, [256], [0, 256])
hist_Berkeley_Aequalizing = cv2.calcHist([Berkeley_Aequalizing], [0], None, [256], [0, 256])

plt.clf()
plt.plot(hist_Berkeley, label = 'original')
plt.xlim([0,256])
plt.legend(loc = 'upper right')
plt.plot(hist_Berkeley_equalizing, label = 'equalized')
plt.xlim([0,256])
plt.legend(loc = 'upper right')
plt.plot(hist_Berkeley_Aequalizing, label = 'Aequalized')
plt.xlim([0,256])
plt.legend(loc = 'upper right')
plt.savefig('Assignment4\\hist_Berkeley.png')