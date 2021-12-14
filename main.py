from dataProcess import *
from baseModel import *
import tensorflow as tf

# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print("Passou\n\n")

    # define model
    model = define_model()

    print("Passou\n\n")

    run_test_harness()
