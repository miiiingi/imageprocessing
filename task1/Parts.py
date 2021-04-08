import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2

## Task 0. Implement a function that convolves image with a given kernel
def conv(image, kernel):

    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image.shape)

    for i in range(h, image_h - h):
        for j in range(w, image_w - w):
            Sum = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    Sum += kernel[m][n] * image[i - h + m][j - w + n]
            image_conv[i][j] = Sum
    return image_conv


# Task 2. Thresholding the edge images.
def thresholding(image):
    value = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            value.append(image[x, y])
    value = pd.Series(sorted(value))
    threshold = value.quantile(q=0.98)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] >= threshold:
                image[x, y] = 255
            else:
                image[x, y] = 0
    image = np.array(image, dtype=np.uint8)
    return image


# Task 3. To convert a gray-scale image to RGB image.
def convert_image(image):
    data = []
    for y in range(image.shape[0]):  # 254
        for x in range(image.shape[1]):  # 668
            if image[y][x] == 255:
                data.append([x, y])
    return data


# Task 3. Perform a RANSAC Algorithm.
def RANSAC(data, iterations=1000, distance_criteria=0.01):
    model = None
    number_inliers = 0

    for i in range(iterations):
        inlier = []

        # Randomly sample a two points 
        sample = random.sample(data, 2)
        x1 = sample[0][0]
        y1 = sample[0][1]
        x2 = sample[1][0]
        y2 = sample[1][1]

        # Solve a linear equation with the picked points 
        try:
            a = (y2 - y1) / (x2 - x1)
            b = -1
            c = a * -x1 + y1

            new_model = [a / c, b / c, 1]

        except ZeroDivisionError:
            continue

        # Calculate a minimum distance using inner product between model vector and all data
        distance = np.sum(new_model * np.pad(data, (0, 1), "constant", constant_values=1), axis=1)


        # If distance is less than distance_criteria, append inlier list.
        for j in range(len(data)):
            if np.abs(distance[j]) < distance_criteria:
                inlier.append(data[j])

        # Compare previous best model's inlier and current model's inlier. If current models's inlier is more than previous best model, reassign current model to best model.
        if number_inliers < len(inlier):
            model = new_model
            number_inliers = len(inlier)

        if (i + 1) % 10 == 0:
            print('epoch : {}'.format(i + 1))

    return model