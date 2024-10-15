import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu_threshold(image):
    # Flatten the image into a 1D array of pixel values (0-255)
    pixel_values = image.ravel()
    
    # Calculate the histogram (256 bins, one for each intensity value)
    hist, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256))
    
    # Total number of pixels
    total_pixels = pixel_values.size
    
    # Initialize variables
    current_max = 0
    threshold = 0
    sum_total = np.dot(np.arange(256), hist)  # Sum of all intensity levels weighted by their frequencies
    sum_background = 0
    weight_background = 0
    weight_foreground = 0
    
    for i in range(256):
        weight_background += hist[i]  # Weight of the background class
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background  # Weight of the foreground class
        if weight_foreground == 0:
            break
        
        sum_background += i * hist[i]  # Sum of background class intensity
        
        mean_background = sum_background / weight_background  # Mean of background class
        mean_foreground = (sum_total - sum_background) / weight_foreground  # Mean of foreground class
        
        # Calculate between-class variance
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Check if the current between-class variance is the largest we've seen
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i
    
    return threshold

# Load a grayscale image
image = cv2.imread('sailboat.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Otsu's algorithm to determine the optimal threshold
optimal_threshold = otsu_threshold(image)

# Threshold the image using the optimal threshold
_, thresholded_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)

# Display the original and thresholded image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title(f'Thresholded Image (Otsu Threshold = {optimal_threshold})')

plt.show()
