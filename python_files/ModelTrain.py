#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:27:44 2021

@author: shirzlotnik
"""


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score
from keras.optimizers import Adam

from keras.preprocessing import image

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import random

import os


import my_Model
import PrintsForUser


class training_Model:
    
    def __init__(self, data, data_train, data_val, data_test):
        self.num_classes = 7
        self.width = 48
        self.height = 48
        self.num_features = 64
        self.batch_size = 512
        self.num_epochs = 3       
        self.data = data
        self.data_train = data_train
        self.data_val   = data_val
        self.data_test  = data_test
        self.model = my_Model.my_model.build_model(self.width, 
                    self.height, self.num_classes, self.num_features)
        
    
    
    
    
    def __CRNO(self, data, dataName):
        """
        (i) convert strings to lists of integers
        (ii) reshape and normalise grayscale image with 255.0
        (iii) one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]
        """
        try:
            data['pixels'] = data['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split(' ')])    
            data_X = np.array(data['pixels'].tolist(), 
                              dtype='float32').reshape(-1,self.width, self.height,1)/255.0   
            data_Y = to_categorical(data['emotion'], self.num_classes)  
            PrintsForUser.print_process("{}, _X shape: {}, , {}, _Y shape: {}".format(dataName, 
                                data_X.shape, dataName, data_Y.shape))
        
        except ValueError:
            PrintsForUser.print_error('okay gurl \n')
        
        return data_X, data_Y 
    
    
    
    def __train_model(self):
        """
        training the model
        """
        PrintsForUser.print_process("[INFO] Training the network...")
        print()
        #
        train_X, train_Y = self.__CRNO(self.data_train, "train") #training data
        val_X, val_Y     = self.__CRNO(self.data_val, "val") #validation data
        test_X, test_Y   = self.__CRNO(self.data_test, "test") #test data
        
        # construct the image generator for data augmentation
        data_generator = ImageDataGenerator(
                                featurewise_center=False,
                                featurewise_std_normalization=False,
                                rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=.1,
                                horizontal_flip=True)
        
        es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', 
                           restore_best_weights=True)
        
        # initialize the model
        PrintsForUser.print_process("[INFO] Compiling model...")
        
        #self.model.compile(loss='categorical_crossentropy', 
         #            optimizer=Adam(lr=0.001, beta_1=0.9, 
          #             beta_2=0.999, epsilon=1e-7), 
           #          metrics=['accuracy'])
        
        self.model.compile(optimizer= 'adam' , loss= 'categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        
        #train the model by slicing the data into "batches" of size batch_size, 
        #and repeatedly iterating over the entire dataset for a given number of epochs.

        history = self.model.fit_generator(data_generator.flow(train_X, train_Y, self.batch_size),
                                      steps_per_epoch=len(train_X) / self.batch_size,
                                      epochs=self.num_epochs,
                                      verbose=2, 
                                      callbacks = [es],
                                      validation_data=(val_X, val_Y))
        
        self.model.save('model.h5')

        return history
    
   
    def __plot_lossAndacc(self, history):
        """
        history: holds a record of the loss values and metric values during training
        plot graph accuracy for epoch and loss for epoch
        """        
        fig, axes = plt.subplots(1,2, figsize=(18, 6))
        # Plot training & validation accuracy values
        axes[0].plot(history.history['accuracy'])
        axes[0].plot(history.history['val_accuracy'])
        axes[0].set_title('Model accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Model loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    
    def handle_train(self):
        """
        this public method manage the train section
        return the train model path and the images labels path
        """
        history = self.__train_model()
        self.__plot_lossAndacc(history)
        self.EvaluateTestPerformance()
        self.predict_images()
        
        
        
    def EvaluateTestPerformance(self):
        """
        evaluating test performance of the model
        """
        test_X, test_Y   = self.__CRNO(self.data_test, "test") #test data

        test_true = np.argmax(test_Y, axis=1)
        test_pred = np.argmax(self.model.predict(test_X), axis=1)
        PrintsForUser.print_process("CNN Model Accuracy on test set: {:.4f}".format(
                accuracy_score(test_true, test_pred)))
        
        
    def predict_images(self):
        """
        predicting 5 first images from the dataset and printing the results
        """
        test_X, test_Y   = self.__CRNO(self.data_test, "test") #test data
            
        
        #predict num_img randomly
        x_pred = self.model.predict(test_X[:5])
        y_res = test_Y[:5]
        
        
        for i in range(5):
            x_class_pred = self.__findIndex_for_maxValuew(x_pred[i])
            y_class_res = self.__findIndex_for_maxValuew(y_res[i])
            
            PrintsForUser.print_process('Model predict -> {}, true result -> {}'.format(
                    x_class_pred, y_class_res))
        
        
        
    def __findIndex_for_maxValuew(self, li):
        """
        the function finds the index that its value is the heighest 
        li: list
        return: int- index of hieghest value
        """
        ind = 0
        max_value = 0
        for info in li:
            if li[info] > max_value:
                max_value = li[info]
                ind = info
                
        return ind

