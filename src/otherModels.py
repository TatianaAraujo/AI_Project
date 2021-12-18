import matplotlib.pyplot as plt
import model
import info


import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img, \
    load_img, save_img


def getNoisyImage(image):
    imageArray = img_to_array(image)
    row, col, ch = imageArray.shape
    mean = 0.9
    var = 0.6
    sigma = var**0.5
    print(image)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    print(gauss)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + (gauss * 100)
    return array_to_img(noisy)