import glob
import numpy as np
from Parts import *
import cv2

# Set a hyper-parameters
distance_criteria = 0.01
# Use a abspath of lane.png
originalimg = glob.glob("lane.png")
originalimg = cv2.imread(originalimg[0], cv2.IMREAD_GRAYSCALE)

# To see a original image.
cv2.imwrite("original.png", originalimg)

# Define a various kernel.
sobelkernel = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3)
laplaciankerenl1 = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape(3, 3)
laplaciankerenl2 = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape(3, 3)
canny_xkernel = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3, 3)
canny_ykernel = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(3, 3)

## Task 1. Detect Edges from the given image (lane.png) using the convolution function you implemented in Task 0
# To show implementing to convolve filter with image, If you want to show a another image, you can change a kernel.
image = conv(originalimg, laplaciankerenl1)
cv2.imwrite("image_edgedetection.png", image)

## Task 2. Thresholding the edge images.
image = thresholding(image)
cv2.imwrite("image_thresholding.png", image)

## Task 3. Perform a RANSAC algorithm to detect lines which constitutes the lane on the Edge Image, and Draw the lines on the image. All the examples below are good results for this task.
# To convert a gray-scale image to RGB image
backgroundimage = np.stack([image, image, image], axis=2)  # BGR channels
image = convert_image(image)
model = RANSAC(image, distance_criteria=distance_criteria)
A, B, C = model

new_data = []
distance = np.sum(model * np.pad(image, (0, 1), "constant", constant_values=1), axis=1)

for i in range(len(image)):
    if np.abs(distance[i]) > distance_criteria:
        new_data.append(image[i])

model = RANSAC(new_data, distance_criteria=distance_criteria)
D, E, F = model

cv2.line(
    backgroundimage,
    (0, int(-C / B)),
    (backgroundimage.shape[1], int(-(C + (backgroundimage.shape[1]) * A) / B)),
    (0, 0, 255),
    1,
)
cv2.line(
    backgroundimage,
    (0, int(-F / E)),
    (backgroundimage.shape[1], int(-(F + (backgroundimage.shape[1]) * D) / E)),
    (0, 0, 255),
    1,
)

cv2.imwrite("image_ransac.png", backgroundimage)