# Do the pyramid representation

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the image
image_path = 'cat.jpg'
image = cv2.imread(image_path)
cv2.imshow('Image', image)
cv2.waitKey(0)

# 1. Downsample the image to 2, 4, 8, 16
scales = [2, 4, 8, 16]

# Nearest neighbor
def downsample_nearest(image, scale):
    return cv2.resize(image, (image.shape[1]//scale, image.shape[0]//scale), interpolation=cv2.INTER_NEAREST)

# Bilinear filtering
def downsample_bilinear(image, scale):
    return cv2.resize(image, (image.shape[1]//scale, image.shape[0]//scale), interpolation=cv2.INTER_LINEAR)

# Max pooling
def downsample_maxpool(image, scale):
    
    ksize = (scale, scale)
    return cv2.resize(cv2.dilate(image, np.ones(ksize)), (image.shape[1]//scale, image.shape[0]//scale))

# 2. Up sample the 1/16 representation to original size

# Bilinear interpolation
def upsample_bilinear(image, original_shape):
    return cv2.resize(image, original_shape, interpolation=cv2.INTER_LINEAR)

# SINC interpolation
def upsample_sinc(image, original_shape):
    return cv2.resize(image, original_shape, interpolation=cv2.INTER_LANCZOS4)



# Downsample and upsample images
scales = [2, 4, 8, 16]
pyramid_nearest = [downsample_nearest(image, scale) for scale in scales]
pyramid_bilinear = [downsample_bilinear(image, scale) for scale in scales]
pyramid_maxpool = [downsample_maxpool(image, scale) for scale in scales]

# Upsample the 1/16 image back to original size
upsampled_bilinear = upsample_bilinear(pyramid_nearest[-1], (image.shape[1], image.shape[0]))
upsampled_sinc = upsample_sinc(pyramid_nearest[-1], (image.shape[1], image.shape[0]))

# Plot the pyramids
fig, axes = plt.subplots(4, len(scales), figsize=(20, 12))

for i, scale in enumerate(scales):
    axes[0, i].imshow(pyramid_nearest[i])
    axes[0, i].set_title(f'Nearest (1/{scale})')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(pyramid_bilinear[i])
    axes[1, i].set_title(f'Bilinear (1/{scale})')
    axes[1, i].axis('off')
    
    axes[2, i].imshow(pyramid_maxpool[i])
    axes[2, i].set_title(f'MaxPool (1/{scale})')
    axes[2, i].axis('off')

# Plot the upsampled images
axes[3, 0].imshow(upsampled_bilinear)
axes[3, 0].set_title('Upsampled Bilinear')
axes[3, 0].axis('off')

axes[3, 1].imshow(upsampled_sinc)
axes[3, 1].set_title('Upsampled SINC')
axes[3, 1].axis('off')

# Hide the last two plots (since we have only 2 upsampled images)
axes[3, 2].axis('off')
axes[3, 3].axis('off')

plt.tight_layout()
plt.show()
