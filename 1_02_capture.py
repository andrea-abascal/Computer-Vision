# Estimate f number of webcam

import cv2
import os
import numpy as np

def measure_object(image, ref_object_width, distance_to_object):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Measure reference object in pixels
    edges = cv2.Canny(gray, 180, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    # Draw the contour on the original image
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green color with thickness 2
    
    # Show the image with the contour
    cv2.imshow('Contour', image)
    cv2.waitKey(0)
    x, y, w, h = cv2.boundingRect(contour)
    object_width_pixels = w
    
    # Compute focal lenght f = z_m*w_pp/w_m
    f = (distance_to_object * object_width_pixels) / ref_object_width
    
    return f, object_width_pixels
    

def estimate_distance(image,f, ref_object_width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Measure reference object in pixels
    edges = cv2.Canny(gray, 250, 300)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    # Draw the contour on the original image
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green color with thickness 2
    # Show the image with the contour
    cv2.imshow('Contour', image)
    cv2.waitKey(0)
    x, y, w, h = cv2.boundingRect(contour)
    object_width_pixels = w
    # Estimate the distance
    d_estimated = (f * ref_object_width) / object_width_pixels
    return d_estimated

# Configuration
ref_object_width = 0.097  # meters (actual size of the reference object)
ref_object_width_test = 0.06
images_folder = 'captured_img'  
images_folder_test = 'captured_img_test' 
distances_file = 'captured_distances.txt' 

# Read the list of distances from the text file
with open(distances_file, 'r') as file:
    distances_to_object = [float(line.strip()) for line in file.readlines()]

# Get the list of image filenames
image_filenames = sorted(os.listdir(images_folder))
image_filenames_test = sorted(os.listdir(images_folder_test))
 
assert len(image_filenames) == len(distances_to_object), "Number of images and distances dont match"

f_values, error_values = [], []
def mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

print("--------------------FOCAL LENGTH CALIBRATION----------------------")

for i, filename in enumerate(image_filenames):
    image_path = os.path.join(images_folder, filename)
    image = cv2.imread(image_path)
    
    # Retrieve distances
    distance_to_object = distances_to_object[i]
    test_image_path = os.path.join(images_folder_test,image_filenames_test[i])
    test_image = cv2.imread(test_image_path)
  
    
    # Measure object pixel and calculate focal length
    f, object_width_pixels = measure_object(image, ref_object_width, distance_to_object)
    print(f"Image: {filename}, Estimated focal length (in pixels): {f}")

    # Check distance
    d_estimated = estimate_distance(test_image,f, ref_object_width_test)
    print(f"Estimated distance for the same image: {d_estimated} meters")

    # Compute mean absolute error 
    error_1 = mae(distance_to_object, d_estimated)*100
    error = abs((d_estimated - distance_to_object) / distance_to_object) * 100
    print(f"Measurement error for image {filename}: {error_1:.2f}%\n")

    f_values.append(round(f,2))
    error_values.append(round(error_1,2))

print(f'Average Focal Length {sum(f_values)/len(f_values)} pixels. MAE: {sum(error_values)/len(error_values)}')
