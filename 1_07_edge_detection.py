import cv2
import numpy as np

import matplotlib.pyplot as plt

image_path = 'cat.jpg'
image = cv2.imread(image_path)
cv2.imshow('Image', image)
cv2.waitKey(0)

#Grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Gradients using Sobel operator
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude and angle of the gradient
magnitude = cv2.magnitude(grad_x, grad_y)
angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

# Normalize magnitude and angle for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
angle = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)

#  RGB image (B:angle, G:gray, R:magnitude)
rgb_image = cv2.merge([angle.astype(np.uint8), image, magnitude.astype(np.uint8)])

plt.figure(figsize=(10, 5))
plt.imshow(rgb_image)
plt.title("RGB Image: Red - Gradient Magnitude, Blue - Gradient Angle, Green - Grayscale")
plt.axis('off')
plt.show()

# Zero-Crossing of Laplacian
laplacian = cv2.Laplacian(image, cv2.CV_64F)
zero_crossing = np.zeros_like(laplacian)

# Detect zero crossings
zero_crossing[np.where((laplacian[:-1, :-1] * laplacian[1:, 1:]) < 0)] = 255
zero_crossing[np.where((laplacian[:-1, 1:] * laplacian[1:, :-1]) < 0)] = 255

plt.figure(figsize=(10, 5))
plt.imshow(zero_crossing, cmap='gray')
plt.title("Zero-Crossing Edges")
plt.axis('off')
plt.show()

# Canny Edge
sigma_values = [0.33, 1.0, 1.5]
fig, axes = plt.subplots(1, len(sigma_values), figsize=(15, 5))

for i, sigma in enumerate(sigma_values):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    canny_edges = cv2.Canny(blurred, 100, 200)

    axes[i].imshow(canny_edges, cmap='gray')
    axes[i].set_title(f'Canny Edges (Sigma={sigma})')
    axes[i].axis('off')

plt.show()