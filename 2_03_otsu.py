# Implement own version of Otsu Algorithm

import cv2 
import numpy as np
from matplotlib import pyplot as plt

# 1. Process the input image
image_path = 'sailboat.jpg'
image = cv2.imread(image_path, 0)
#image = cv2.GaussianBlur(image, (5, 5), 0)

plt.hist(image.ravel(), 256)
plt.xlabel('Colour intensity')
plt.ylabel('Number of pixels')
plt.savefig("otsu_image_hist.png")
plt.show()

# 2. Obtain image histogram (distribution of pixels)
def otsu_threshold(image):
    """
    Applies Otsu's method to find the optimal threshold for binarizing an image.

    Args:
      image: A grayscale image.

    Returns:
      Otsu optimal threshold value.
    """

    bins_num = 256 # (256 bins, one for each intensity value)
    hist, bin_edges = np.histogram(image, bins=bins_num)

    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2. # Calculate centers of bins

    # 3. Compute the threshold value T using OTSU algorithm

    weight1 = np.cumsum(hist) # get the probabilities w1(t) - background 0
    weight2 = np.cumsum(hist[::-1])[::-1] # get the probabilities  w2(t) - foreground 1
    
    mean1 = np.cumsum(hist * bin_mids) / weight1 # Get the class means mu0(t) - background 0
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1] # Get the class means mu1(t) - foreground 1
    
    between_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    index_of_max_val = np.argmax(between_class_variance) # Find the idx of the max in between_class_variance function val
    
    threshold = bin_mids[:-1][index_of_max_val]
    print("Otsu's algorithm implementation thresholding result: ", threshold)

    return round(threshold,2)

# 4. Replace image pixels into white in those regions, where saturation is greater than T and into the black in the opposite cases.
def imp_otsu_threshold(image,threshold):
    # Threshold the image using the computed threshold
    thresholded_image = np.zeros_like(image)

    # Manual thresholding
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= threshold:
                thresholded_image[i, j] = 255  # Set foreground to white
            else:
                thresholded_image[i, j] = 0    # Set background to black
    return thresholded_image


def cv2_otsu_threshold(image):
    cv2_otsu_t, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
    return cv2_otsu_t,thresholded_image

# 5. Compare and Visualize
otsu_t = otsu_threshold(image)
img = imp_otsu_threshold(image,otsu_t)
cv2.imwrite('otsu_image.jpg',img)
cv2_t, cv2_img = cv2_otsu_threshold(image)

# Plot histogram foreground and backgroudn classes based on otsu threshold
pixel_values = image.ravel()
plt.hist(pixel_values[pixel_values < otsu_t], bins=256, color='red', label='Background')
plt.hist(pixel_values[pixel_values >= otsu_t], bins=256, color='blue', label='Foreground')
plt.xlabel('Colour intensity')
plt.ylabel('Number of pixels')
plt.legend()
plt.savefig("otsu_hist_result.png")

# Display the original and thresholded image
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image', fontsize=8)

plt.subplot(1, 3, 2)
plt.imshow(img, cmap='gray')
plt.title(f'Otsu Threshold (implementation) = {otsu_t}', fontsize=8)

plt.subplot(1, 3, 3)
plt.imshow(cv2_img, cmap='gray')
plt.title(f'Otsu Threshold (OpenCV) = {cv2_t}', fontsize=8)
plt.savefig("otsu_results.png")
plt.show()