
import sys
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import info


def define_model_one_block_vgg():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def define_model_two_block_vgg():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same',
              input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu',
              kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def define_model_three_block_vgg():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def define_model_dropout():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# plot diagnostic learning curves


def summarize_diagnostics(history):
    print("Plot!!!!!\n\n")
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(info.plotDir + sys.argv[1] + '.png')
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    if (len(sys.argv) < 1):
        print("Argument error")
        return

    # Chose from different models
    if (sys.argv[1] == "vgg1"):
        model = define_model_one_block_vgg()

    elif(sys.argv[1] == "vgg2"):
        model = define_model_two_block_vgg()

    elif(sys.argv[1] == "vgg3"):
        model = define_model_three_block_vgg()

    else:
        model = define_model_dropout()

    # Chose from normal train set or train set with noise images
    if (len(sys.argv) > 2 and sys.argv[2] == "imgAgu"):
        datagen = ImageDataGenerator(rescale=1.0/255.0,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True)
        sourceDir = info.dataDir + "trainNoise/"

    else:
        datagen = ImageDataGenerator(rescale=1.0/255.0)
        sourceDir = info.dataDir + "train/"

    # prepare iterators
    train_it = datagen.flow_from_directory(sourceDir,
                                           class_mode='binary',
                                           batch_size=info.batchNumber,
                                           target_size=(200, 200))

    test_it = datagen.flow_from_directory(info.dataDir + 'test/',
                                          class_mode='binary',
                                          batch_size=info.batchNumber,
                                          target_size=(200, 200))

    print(sourceDir)

    history = model.fit_generator(train_it,
                                  steps_per_epoch=len(train_it),
                                  validation_data=test_it,
                                  validation_steps=len(test_it),
                                  epochs=20, verbose=1)

    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))

    modelPath = "models/" + sys.argv[1]

    if (len(sys.argv) > 2):
        modelPath += "." + sys.argv[2]

    model.save(modelPath)

    # learning curves
    summarize_diagnostics(history)
