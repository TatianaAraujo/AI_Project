# plot dog photos from the dogs vs cats dataset
from genericpath import exists
from matplotlib import pyplot
from matplotlib.image import imread
import os
from pathlib import Path
from shutil import copyfile
import random as rnd
import info
import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img, load_img, save_img
import matplotlib.pyplot as plt

def testDogPlot():
    # define location of dataset
    folder = 'data/train/'
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder + 'dog.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()

def testCatPlot():
    folder = 'data/train/'
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder + 'cat.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()

def folderCreation():
    print("Creating folders")
    # create directories
    dataset_home = info.dataDir
    subdirs = ['train/', 'test/', 'trainNoise/']
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)
        # create label subdirectories
        labeldirs = ['dogs/', 'cats/']
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            os.makedirs(newdir, exist_ok=True)

def checkData():
    if ( os.path.exists(info.dataDir) == False):
        folderCreation()
    
    if( os.path.exists(info.dataDir + "trainNoise") == False):
        folderCreation()

    if ( os.path.exists(info.dataDir + "/trainNoise/" +"cats" + "cat.1.jpg") == False):
        createWhiteNoite()

def dataSelection():
    rnd.seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25

    dataset_home = info.dataDir

    # copy training dataset images into subdirectories
    src_train_directory = info.rawDataDir + '/train/'

    for file in os.listdir(src_train_directory):
        src = src_train_directory + '/' + file
        dst_dir = 'train/'
        if rnd.random() < val_ratio:
            dst_dir = 'test/'
        if file.startswith('cat'):
            dst = dataset_home + dst_dir + 'cats/'  + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dataset_home + dst_dir + 'dogs/'  + file
            copyfile(src, dst)

def createWhiteNoite():
    # create directories
    dir = info.dataDir + "/train/"

    for file in os.listdir(dir):
        print(file)

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


