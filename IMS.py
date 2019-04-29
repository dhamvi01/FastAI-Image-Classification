import os
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
path = "C:/Users/vijaykumar.dhameliya/Desktop/1000P/19 Image classification/dogscats/dogscats"
from keras.preprocessing.image import img_to_array
import keras

cat = os.listdir(path + '/train/cats/')
os.chdir(path + '/train/cats/')
data = []
labels = []
for i in cat:
    print(i)
    img = cv2.imread(i)
    img = cv2.resize(img, (300,300))
    img = img_to_array(img)
    data.append(img)
    labels.append(0)
                 
cat = os.listdir(path + '/train/dogs/')
os.chdir(path + '/train/dogs/')
for i in cat:
    print(i)
    img = cv2.imread(i)
    img = cv2.resize(img, (300,300))
    img = img_to_array(img)
    data.append(img)
    labels.append(1)

filters = 10
filter_size = (5,5)

epoch = 5
b_size  =128
input_size = (300,300,3)

from keras.utils.np_utils import to_categorical
labels = to_categorical(labels)

data = np.array(data, dtype="float")/ 255.0
labels = np.array(labels)


model = Sequential()
model.add(keras.layers.InputLayer(input_shape=input_size))

model.add(keras.layers.convolutional.Conv2D(filters, filter_size, strides=(1,1), padding='valid', data_format='channels_last',activation='relu' ))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=2, input_dim = 50, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
printmodel.fit(data, labels, epochs=epoch, batch_size=b_size, validation_split=0.3)

print(model.summary())
              


















