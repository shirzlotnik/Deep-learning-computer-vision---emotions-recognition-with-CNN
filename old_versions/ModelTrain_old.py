#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:27:44 2021

@author: shirzlotnik
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator



import my_Model
#import unload_dataset
#import preprocessing

class training_Model:
    
    def __init__(self, data, data_train, data_val, data_test):
        self.num_classes = 7
        self.width = 48
        self.height = 48
        self.num_features = 64
        self.batch_size = 64
        self.num_epochs = 64
        #import dataset values from modules
        """
        self.data = unload_dataset.data
        self.data_train = preprocessing.data_train
        self.data_val   = preprocessing.data_val
        self.data_test  = preprocessing.data_test
        """
        self.data = data
        self.data_train = data_train
        self.data_val   = data_val
        self.data_test  = data_test
        
        
    def __CRNO(self, data, dataName):
        """
        (i) convert strings to lists of integers
        (ii) reshape and normalise grayscale image with 255.0
        (iii) one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]
        """
        data['pixels'] = data['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
        data_X = np.array(data['pixels'].tolist(), dtype='float32').reshape(-1,self.width, self.height,1)/255.0   
        data_Y = to_categorical(data['expression'], self.num_classes)  
        print("{}, _X shape: {}, , {}, _Y shape: {}".format(dataName, data_X.shape, dataName, data_Y.shape))
        
        return data_X, data_Y 
    
    
    
    def __train_model(self):
        """
        training the model
        """
        
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
        
        es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)
        
        # initialize the model
        print("[INFO] Compiling model...")
        
        model = my_Model.my_model.build_model(self.width,self.height,self.num_classes,self.num_features)
        
        #model.compile(optimizer= 'adam' , loss= keras.losses.binary_crossentropy, metrics=['accuracy'])
        model.compile(optimizer= 'adam' , loss= 'categorical_crossentropy', metrics=['accuracy'])
        #model.summry()
        model.summary()
        
        
        # train the network
        print("[INFO] Training the network...")
        print()
        
        history = model.fit_generator(data_generator.flow(train_X, train_Y, self.batch_size),
                                      steps_per_epoch=len(train_X) / self.batch_size,
                                      epochs=self.num_epochs,
                                      verbose=2, 
                                      callbacks = [es],
                                      validation_data=(val_X, val_Y))
        
        # Evaluate the model on the test data using `evaluate`
        print('Evaluate on test data')
        
        results = model.evaluate(test_X, test_Y, batch_size=32)
        print('test loss ' + str(results[0])  + ' , test acc ' + str(results[1])) 
        
        model_path = '/Users/shirzlotnik/Desktop'
        
        # save the model to desktop
        print('[INFO] Serializing network...')
        model.save(model_path)
        
        return history
    
    def __plot_graphs(self, history):
        """
        
        """
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='orange', label='test')
        
	    # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(history.history['accuracy'], color='blue', label='train')
        plt.plot(history.history['val_accuracy'], color='orange', label='test')
        
    def __plot_lossAndacc(self, history):
        """
        history:
        plot graph accuracy for epoch and loss for epoch
        """
        fig, axes = plt.subplots(1,2, figsize=(18, 6))
        # Plot training & validation accuracy values
        axes[0].plot(history.history['accuracy'])
        axes[0].plot(history.history['Validation_accuracy'])
        axes[0].set_title('Model accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['Validation_loss'])
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
        self.__plot_graphs(history)
        
        
        
