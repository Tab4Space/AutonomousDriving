from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.preprocessing import image
from keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_plots(history):
    fig = plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train, Test'], loc=0)

    plt.subplot(122)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train, Test'], loc=0)

    plt.savefig('./result.png')

def draw_plots2(history):
    fig = plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.subplot(122)
    plt.plot(history.history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train, Test'], loc=0)

    plt.savefig('./result2.png')

# @profile
def my_func():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(144,256,3)))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(adam, loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()


    train_datagen = ImageDataGenerator(rescale=1./255.)
    train_generator = train_datagen.flow_from_directory(
        directory='./data/class_data/',
        target_size=(144, 256),
        batch_size=32,
        class_mode='categorical'
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 2300//32,
        epochs = 1
    )

    model.save('./models/class2.h5')


if __name__ == '__main__':
    my_func()