from dataProcess import *
from model import *
import tensorflow as tf
import argparse


def generateModels(arguments:argparse.ArgumentParser):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    run_test_harness(arguments)


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs=1,
                        help="The model name to use in keras")
    parser.add_argument("--imgAgu", action='store_true',
                        help="If present then we use images with noise")
    parser.add_argument("--pandas", action='store_true',
                        help="If present we use the dataset with pandas")
    parser.add_argument("--onlyCreateDir", action='store_true',
                        help="if present, only creates the data directories")

    return parser.parse_args()


if __name__ == '__main__':
    parse = parseArguments()

    checkData()

    if ( parse.model != None ):
        generateModels(parse)
