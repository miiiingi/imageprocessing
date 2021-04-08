import cv2
import matplotlib.pyplot as plt
import numpy as np 
import os
# Q1. Convert the image 'KNU_original.jpg' to 'KNU_final.jpg'. Explain details.

# Make a directory
if os.path.isdir('Assignment1') == False :
    os.makedirs('Assignment1')

# Read a Image
image_original = cv2.imread('Images\\KNU_original.jpg')
image_target = cv2.imread('Images\\KNU_final.jpg')
image_target_gray = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)

# Half a image
image_original_left = image_original[:,:948//2+1] 
image_original_right = image_original[:,948//2+1:]

image_target_left = image_target[:,:948//2+1] 
image_target_right = image_target[:,948//2+1:]

image_target_gray_left = image_target_gray[:,:948//2+1] 
image_target_gray_right = image_target_gray[:,948//2+1:]

def calculate_cdf(histogram):
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
 
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    # To make two histogram be similar, Compare the cdf values.
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table

def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)
 
    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])    
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0,256])    
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0,256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0,256])
 
    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)
 
    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)
 
    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)
 
    # Put the image back together
    image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)
    image_after_matching = cv2.cvtColor(image_after_matching, cv2.COLOR_BGR2GRAY)
 
    return image_after_matching

# implement a equalizing histogram
equalized_left = match_histograms(image_original_left, image_target_left)
equalized_right = match_histograms(image_original_right, image_target_right)

# Calibrate a intensity
equalized_left = cv2.convertScaleAbs(equalized_left, beta=-15)
equalized_right = cv2.convertScaleAbs(equalized_right, beta=-15)

plt.clf()
image_concat = np.concatenate((equalized_left, equalized_right, image_target_gray_left, image_target_gray_right), axis=1)
cv2.imwrite('Assignment1\\image_concat.png', image_concat)

# Make a Intensity Histogram
plt.clf()
hist_target_left = cv2.calcHist([image_target_left], [0], None, [256], [0, 256])
hist_equalized_left = cv2.calcHist([equalized_left], [0], None, [256], [0, 256])
plt.xlim([0,256])

plt.plot(hist_target_left, label = 'target')
plt.legend(loc = 'upper right')
plt.plot(hist_equalized_left, label = 'equalized')
plt.legend(loc = 'upper right')
plt.savefig('Assignment1\\hist_left.png')

# Make a Intensity Histogram
plt.clf()
hist_target_right = cv2.calcHist([image_target_right], [0], None, [256], [0, 256])
hist_equalized_right = cv2.calcHist([equalized_right], [0], None, [256], [0, 256])
plt.xlim([0,256])

plt.plot(hist_target_right, label = 'target')
plt.legend(loc = 'upper right')
plt.plot(hist_equalized_right, label = 'equalized')
plt.legend(loc = 'upper right')
plt.savefig('Assignment1\\hist_right.png')