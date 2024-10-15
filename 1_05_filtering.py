# Test different filters

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the image
image_path = 'cat.jpg'
image = cv2.imread(image_path)
cv2.imshow('Image', image)
cv2.waitKey(0)


# 1. Gaussian Noise

def add_gaussian_noise(img, mean=0, var=0.01):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).astype('float32')
    noisy = cv2.addWeighted(img.astype('float32'), 1.0, gauss, 1.0, 0.0)
    return np.clip(noisy, 0, 255).astype('uint8')

gaussian_noisy_image = add_gaussian_noise(image)

# 2. Salt and Pepper Noise
def add_salt_and_pepper_noise(img, salt_prob=0.05, pepper_prob=0.05):
        noisy = img.copy()
        total_pixels = img.size
        salt_pixels = int(salt_prob * total_pixels)
        pepper_pixels = int(pepper_prob * total_pixels)

        # Salt noise (white pixels)
        coords = [np.random.randint(0, i - 1, salt_pixels) for i in img.shape[:2]]
        noisy[coords[0], coords[1], :] = 255

        # Pepper noise (black pixels)
        coords = [np.random.randint(0, i - 1, pepper_pixels) for i in img.shape[:2]]
        noisy[coords[0], coords[1], :] = 0

        return noisy

sp_noisy_image = add_salt_and_pepper_noise(image)

noisy_images = [gaussian_noisy_image, sp_noisy_image]


# Initialize a figure for plotting
fig, axes = plt.subplots(len(noisy_images), 4, figsize=(15, 10))
subtitles = ['Gaussian Noise', 'Salt and Pepper Noise']
titles = ['Noisy Image', 'Mean Filter', 'Median Filter', 'Anisotropic Filter']

for i, (img, subtitle) in enumerate(zip(noisy_images, subtitles)):
    # Apply filters
    mean_filtered = cv2.blur(img, (5, 5))
    median_filtered = cv2.medianBlur(img, 5)
    anisotropic_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Plot the original and filtered images
    axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i, 1].imshow(cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2RGB))
    axes[i, 2].imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
    axes[i, 3].imshow(cv2.cvtColor(anisotropic_filtered, cv2.COLOR_BGR2RGB))

    # Set titles for each filter
    for j in range(4):
        axes[i, j].set_title(titles[j])
        axes[i, j].axis('off')
    # Add subtitle for the current row
    axes[i, 0].annotate(subtitle, xy=(0, 0), xytext=(-axes[i, 0].xaxis.labelpad - 5, 0),
                        xycoords=axes[i, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='center', va='baseline', rotation=90)

plt.tight_layout()
plt.subplots_adjust(left=0.1, top=0.9) 
plt.show()

# 3. Laplacian Noise
laplacian = cv2.Laplacian(image,cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
cv2.imshow('Laplacian Noise', laplacian)
cv2.waitKey(0)

# 4. Sobel X Noise
sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
sobelx = np.uint8(np.absolute(sobelx))
cv2.imshow('Sobel X Noise', sobelx)
cv2.waitKey(0)

# 5. Laplacian of Gaussian Noise
log = cv2.GaussianBlur(image, (3, 3), 0)
log = cv2.Laplacian(log, cv2.CV_64F)
log = np.uint8(np.absolute(log))
cv2.imshow('Laplacian of Gaussian Noise', log)
cv2.waitKey(0)

# 6. Sharpening Noise
sharpening_kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
sharpened = cv2.filter2D(image, -1, sharpening_kernel)
cv2.imshow('Sharpening Noise', sharpened)
cv2.waitKey(0)

# 7. Homomorphic Filter

# Convolutions Domain

# Step 1: Convert to logarithmic domain
image_log = np.log1p(np.array(image, dtype="float"))

# Step 2: Apply Gaussian filter (simulates convolution with high-pass filter)
# Create a high-pass filter by subtracting a low-pass filter from the identity
gaussian_filter = cv2.GaussianBlur(image_log, (15, 15), 30)
high_pass_filter = image_log - gaussian_filter

# Step 3: Convert back to spatial domain
image_homomorphic = np.expm1(high_pass_filter)
image_homomorphic = np.uint8(cv2.normalize(image_homomorphic, None, 0, 255, cv2.NORM_MINMAX))

cv2.imshow('Homomorphic Filtered Convolution', image_homomorphic)
cv2.waitKey(0)

# Fourier Domain

# Convert to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Convert to logarithmic domain
image_log = np.log1p(np.array(image_gray, dtype="float"))

# Step 2: Fourier Transform
image_fft = np.fft.fftshift(np.fft.fft2(image_log)) 

# Step 3: Create a high-pass filter in the frequency domain
rows, cols = image_gray.shape
crow, ccol = rows // 2, cols // 2

# Create a mask with high-pass filter characteristics
mask = np.ones((rows, cols), np.float32)
r = 30  # Radius of the low-frequency drop-off
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= r
mask[mask_area] = 0

# Apply the mask (high-pass filter) to the Fourier transformed image
image_fft_filtered = image_fft * mask

# Step 4: Inverse Fourier Transform
image_filtered_log = np.abs(np.fft.ifft2(np.fft.ifftshift(image_fft_filtered)))

# Step 5: Convert back to spatial domain
image_homomorphic = np.expm1(image_filtered_log)
image_homomorphic = np.uint8(cv2.normalize(image_homomorphic, None, 0, 255, cv2.NORM_MINMAX))

# Display the result
cv2.imshow('Homomorphic Filtered Fourier', image_homomorphic)
cv2.waitKey(0)

cv2.destroyAllWindows()