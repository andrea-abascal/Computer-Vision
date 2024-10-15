import cv2

# Define the path to the image
image_path = 'cat.jpg'
image = cv2.imread(image_path)

# 1. Capture Image
cv2.imshow('Image', image)
cv2.waitKey(0)

# 2. Swap Red and Green Channels
swapped_image = image.copy()
swapped_image[:, :, 1] = image[:, :, 2]  # Set green channel to red channel
swapped_image[:, :, 2] = image[:, :, 1]  # Set red channel to blue channel
cv2.imshow('Swapped Red and Green',swapped_image)
cv2.waitKey(0)

# 3. Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image',gray)
cv2.waitKey(0)

# 4. Invert Grayscale
inverted_gray = 255 - gray
cv2.imshow('Inverted Grayscale Image',inverted_gray)
cv2.waitKey(0)

# 5. Change to HSV and display only H channel
hsv_ = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h_channel = hsv_[:,:,0]
cv2.imshow('H Channel',h_channel)
cv2.waitKey(0)

cv2.destroyAllWindows()