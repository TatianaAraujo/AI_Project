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


def checkData():
    if (os.path.exists(info.dataDir) == False):
        print("data directory doesn't exist")
        folderCreation()

    if (not os.listdir(info.dataDir + "train/")):
        print("No default dataset")
        dataSelection()

    if(os.path.exists(info.dataDir + "trainNoise") == False):
        print("trainNoise directory doesn't exist")
        folderCreation()

    if (not os.listdir(info.dataDir + "trainNoise/" + "cats/")):
        print("trainNoise directory doesn't cointain images")
        #createWhiteNoite()

    if ( os.path.exists(info.modelDir) == False):
        print("Creating models directory")
        os.makedirs(info.modelDir, exist_ok=True)

    if ( os.path.exists(info.logDir) == False):
        print("Creating log directory")
        os.makedirs(info.logDir, exist_ok=True)

    if ( os.path.exists(info.plotDir) == False):
        print("Creating log directory")
        os.makedirs(info.plotDir, exist_ok=True)

def folderCreation():
    print("Creating folders")

    # create directories
    dataset_home = info.dataDir
    subdirs = ['train/', 'test/', 'trainNoise/']

    for subdir in subdirs:
        os.makedirs(dataset_home + subdir, exist_ok=True)

        # create label subdirectories
        for labldir in ['dogs/', 'cats/']:
            newdir = dataset_home + subdir + labldir
            os.makedirs(newdir, exist_ok=True)

def dataSelection():
    print("Creating default dataset")
    rnd.seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.25

    dataset_home = info.dataDir

    # copy training dataset images into subdirectories
    src_train_directory = info.rawDataDir

    for file in os.listdir(src_train_directory):
        src = src_train_directory + '/' + file

        if rnd.random() < val_ratio:
            dst_dir = 'test/'
        else:
            dst_dir = 'train/'

        if file.startswith('cat'):
            dst = dataset_home + dst_dir + 'cats/' + file
            copyfile(src, dst)
        elif file.startswith('dog'):
            dst = dataset_home + dst_dir + 'dogs/' + file
            copyfile(src, dst)


def createWhiteNoite():
    print("Creating white noise images")

    # create directories
    sourceDir = info.dataDir + "train/"
    destDir = info.dataDir + "trainNoise/"

    for subDir in os.listdir(sourceDir):
        for file in os.listdir(sourceDir + subDir):
            copyfile(sourceDir + subDir + "/" + file, destDir + subDir + "/" + file)
            image = load_img(sourceDir + subDir + "/" + file)
            save_img(destDir + subDir + "/" + "noise." + file, getNoisyImage(image))

def getNoisyImage(image):
    imageArray = img_to_array(image)
    row, col, ch = imageArray.shape
    mean = 0.9
    var = 0.6
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + (gauss * 100)
    return array_to_img(noisy)
