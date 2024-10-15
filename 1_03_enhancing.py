# Do histogram equalization

import cv2

# Define the path to the image
image_path = 'cat.jpg'
image = cv2.imread(image_path)
cv2.imshow('Image', image)
cv2.waitKey(0)

# 1. Apply Equalization on each channel independently
channels = cv2.split(image)
eq_channels = []
for ch in channels:
    eq_channels.append(cv2.equalizeHist(ch))
eq_image = cv2.merge(eq_channels)
cv2.imshow('Equalized Image', eq_image)
cv2.waitKey(0)

# 2. Apply Equalization to the V Channel
hsv_ = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
v_channel = hsv_[:,:,2]
eq_v_channel = cv2.equalizeHist(hsv_[:,:,2])
hsv_[:,:,2] = eq_v_channel
color_image = cv2.cvtColor(hsv_, cv2.COLOR_HSV2BGR)
cv2.imshow('Equalized HSV Image',color_image)
cv2.waitKey(0)

cv2.destroyAllWindows()


# Apply Gamma Enhancing
import numpy as np

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# 1. Enhance low-intensity areas (Gamma > 1)
gamma_low = adjust_gamma(image, gamma=2.0)
cv2.imshow('Low Gamma', gamma_low)
cv2.waitKey(0)

# 2. Enhance high-intensity areas (Gamma < 1)
gamma_high = adjust_gamma(image, gamma=0.5)
cv2.imshow('High Gamma', gamma_high)
cv2.waitKey(0)

cv2.destroyAllWindows()