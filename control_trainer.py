import os
import numpy as np
import pandas as pd

from PIL import Image
from preprocessor import makeCSV
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 1024
CSVFILE = './right.csv'
DATADIRPATH = './_data/'
DATALENGTH = 2003

def custom_generator(dataset_path, bs, useROI=False):
    while True:
        imageList = []
        steerList = []
        cmdList = []

        with open(dataset_path) as f:
            while len(imageList) < bs:
                imagePath, steer = f.readline().split(',')
                
                image = np.array(Image.open(imagePath)).astype(np.float32)
                if useROI is False:
                    image = image[:, :, 0:3]
                else:
                    image = image[78:144, 27:227, 0:3]

                image /= 255
                steer = np.array(np.float32(steer))

                imageList.append(image)
                steerList.append(steer)
            yield({'conv2d_1_input':np.array(imageList)}, {'output':np.array(steerList)})

def build_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(Flatten())    
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='output'))
    model.compile(
        optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss='mse',
        metrics=['accuracy']
    )
    model.summary()
    return model

def main():
    if not os.path.exists(CSVFILE):
        makeCSV(DATADIRPATH, CSVFILE)
        print('make csv file')

    model = build_model()
    model.compile(
        optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss='mse',
        metrics=['mean_absolute_error']
    )

    history = model.fit_generator(
        generator=custom_generator(CSVFILE, BATCH_SIZE, useROI=True),
        steps_per_epoch=DATALENGTH // BATCH_SIZE,
        epochs=100,
        shuffle=False
    )

    model.save('fullPath_003data_004data.h5')


if __name__ == '__main__':
    main()