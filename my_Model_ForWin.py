#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:17:12 2021

@author: shirzlotnik
"""

"""
this file contain a class with a static method
this method declared the model layers
"""

"""
Dropout is a technique where randomly selected neurons are ignored during training. 
They are “dropped-out” randomly. This means that their contribution to the activation 
of downstream neurons is temporally removed on the forward pass and any weight updates 
are not applied to the neuron on the backward pass.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
#from keras.optimizers import adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, InputLayer, Dense, Activation








"""
num_classes = 7 
width, height = 48, 48
num_features = 64
"""
class my_model:
    @staticmethod
    def build_model(width, height, num_classes, num_features):
        """
        Building CNN Model
        CNN Architecture:
            Conv => BN => Activation => Conv => BN => Activation => MaxPooling
            Conv => BN => Activation -> Conv => BN -> Activation => MaxPooling
            Conv => BN => Activation -> Conv => BN -> Activation => MaxPooling
            Flatten
            Dense => BN => Activation
            Dense => BN => Activation
            Dense => BN => Activation
            Output layer
        """
        
        model = keras.Sequential()
        
        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=(width, height, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(2*num_features, kernel_size=(3, 3), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        #FC => RELU layers
        model.add(Flatten())
        # dense 1
        model.add(Dense(2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # dense 2
        model.add(Dense(2*2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # dense 3
        model.add(Dense(2*2*2*num_features))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        # softmax classifier
        model.add(Dense(num_classes, activation='softmax'))
        

        return model
    

    
        
        
        
