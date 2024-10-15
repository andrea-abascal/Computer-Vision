# Implement IIF and FIR 
import cv2
import numpy as np
import scipy
from scipy.signal import lfilter, iirfilter, convolve2d, firwin

# Define the path to the image
image_path = 'cat.jpg'
image = cv2.imread(image_path)

# 1. Capture Image
cv2.imshow('Image', image)
cv2.waitKey(0)

# Define an IIR filter (Butterworth filter)
b, a = iirfilter(4, 0.2, btype='low', analog=False, ftype='butter')

# Apply the IIR filter to the image
image_iir_filtered = lfilter(b, a, image, axis=0)  # Apply along rows
image_iir_filtered = lfilter(b, a, image_iir_filtered, axis=1)  # Apply along columns
image_iir_filtered = np.uint8(image_iir_filtered)

# Display the result
cv2.imshow('IIR Filtered', image_iir_filtered)

# Define an FIR filter (Low-pass filter)
fir_coeff = firwin(21, 0.2, window='hamming')

# Apply the FIR filter to the image using 2D convolution
image_fir_filtered = convolve2d(image, fir_coeff[:, None], mode='same', boundary='wrap')
image_fir_filtered = convolve2d(image_fir_filtered, fir_coeff[None, :], mode='same', boundary='wrap')

# Normalize the filtered image to the range [0, 255] and convert to uint8
image_fir_filtered = cv2.normalize(image_fir_filtered, None, 0, 255, cv2.NORM_MINMAX)
image_fir_filtered = np.uint8(image_fir_filtered)

# Display the result
cv2.imshow('FIR Filtered', image_fir_filtered)
