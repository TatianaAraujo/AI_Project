# importing the required libraries
import numpy as np
import skimage
from skimage.io import imread, imshow
from skimage.filters import *
import matplotlib.pyplot as plt
import skimage.filters

# matplotlib inline

# reading the image
image = imread('data/train/cats/cat.2.jpg', as_gray=True)

# calculating horizontal edges using prewitt kernel
edges_prewitt_horizontal = prewitt_h(image)
print(edges_prewitt_horizontal)
# calculating vertical edges using prewitt kernel
edges_prewitt_vertical = prewitt_v(image)
print(edges_prewitt_vertical)

for l in range(len(edges_prewitt_horizontal)):
    for c in range(len(edges_prewitt_horizontal[l])):
        if(edges_prewitt_horizontal[l][c] > 4):
            edges_prewitt_horizontal[l][c] -= 6
            print("olá")

imshow(edges_prewitt_horizontal, cmap='gray')
plt.show()
