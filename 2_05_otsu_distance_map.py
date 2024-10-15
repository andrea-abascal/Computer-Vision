import cv2
from scipy import ndimage

image = cv2.imread('otsu_image.jpg')
# Invert the binary image (so the object is black, background is white)
inverted_binary = cv2.bitwise_not(image)

# Compute the distance transform (distance map)
distance_map = ndimage.distance_transform_edt(inverted_binary)
# Normalize the distance map for visualization
distance_map_normalized = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# Show the distance map
cv2.imshow('Distance Map', distance_map_normalized)
cv2.waitKey(0)