#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:07:13 2021

@author: shirzlotnik
"""


"""
this python file handle the train section
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from keras.optimizers import adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping



import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


import my_Model
import unload_dataset
import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os

model_path = '/Users/shirzlotnik/Desktop'


data = unload_dataset.data
data_train = preprocessing.data_train
data_val   = preprocessing.data_val
data_test  = preprocessing.data_test


#initilize parameters
num_classes = 7 
width, height = 48, 48
num_epochs = 50
batch_size = 64
num_features = 64

"""
CRNO stands for Convert, Reshape, Normalize, One-hot encoding
(i) convert strings to lists of integers
(ii) reshape and normalise grayscale image with 255.0
(iii) one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]
"""

def CRNO(data, dataName):
    data['pixels'] = data['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
    data_X = np.array(data['pixels'].tolist(), dtype='float32').reshape(-1,width, height,1)/255.0   
    data_Y = to_categorical(data['emotion'], num_classes)  
    print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
    return data_X, data_Y

    
train_X, train_Y = CRNO(data_train, "train") #training data
val_X, val_Y     = CRNO(data_val, "val") #validation data
test_X, test_Y   = CRNO(data_test, "test") #test data


def train_model():
    """
    
    """

    # construct the image generator for data augmentation
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)
    
    es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)
    
    # initialize the model
    print("[INFO] Compiling model...")
    
    model = my_Model.my_model.build_model(width,height,num_classes,num_features)
    
    model.compile(optimizer= 'adam' , loss= keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.summry()
    
    
    # train the network
    print("[INFO] Training the network...")
    print()
    
    history = model.fit_generator(data_generator.flow(train_X, train_Y, batch_size),
                                  steps_per_epoch=len(train_X) / batch_size,
                                  epochs=num_epochs,
                                  verbose=2, 
                                  callbacks = [es],
                                  validation_data=(val_X, val_Y))
    
    # Evaluate the model on the test data using `evaluate`
    print('Evaluate on test data')
    
    results = model.evaluate(test_X, test_Y, batch_size=32)
    print('test loss ' + str(results[0])  + ' , test acc ' + str(results[1])) 
    
    # save the model to desktop
    print('[INFO] Serializing network...')
    model.save(model_path)
    
    return history
    
