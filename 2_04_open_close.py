import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = np.ones((3,3), np.uint8)

def opening(img):
    erosion = cv2.erode(img,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    return dilation

def closing(img):
    dilation = cv2.dilate(img,kernel,iterations = 1)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    return erosion

def clean(img_path,method):
  img = cv2.imread(img_path)
  open = opening(img)
  close = closing(img)

  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(img, cmap='gray')
  plt.title(f'{method} Opening', fontsize=8)
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(img, cmap='gray')
  plt.title(f'{method} Closing', fontsize=8)
  plt.axis('off')

  plt.savefig(f"{method}_morphology.png")
  plt.show()

"""
#-------------REGION GROWING ALGORITHM-----------------

rg_image_path = 'region_g_results.png'
method = 'Region Growing'
clean(rg_image_path, method)

#-------------K MEANS ALGORITHM-----------------

kmeans_image_path = 'kmeans_results.png'
method = 'K Means'
clean(kmeans_image_path, method)

#-------------FUZZY C MEANS ALGORITHM-----------------

fuzzyc_image_path = 'fuzzyc_results.png'
method = 'Fuzzy C Means'
clean(fuzzyc_image_path, method)
"""
#-------------OTSU ALGORITHM-----------------

otsu_image_path = 'otsu_image.jpg'
method = 'Otsu'
clean(otsu_image_path, method)
