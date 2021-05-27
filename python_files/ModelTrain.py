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

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import random

from sklearn.model_selection import train_test_split


from keras.models import Sequential


import my_Model
import unload_dataset
#import preprocessing

class training_Model:
    
    def __init__(self, data, data_train, data_val, data_test):
        self.num_classes = 7
        self.width = 48
        self.height = 48
        self.num_features = 64
        self.batch_size = 128
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
        self.model = keras.Sequential()
        
        
    def __CRNO(self, data, dataName):
        """
        (i) convert strings to lists of integers
        (ii) reshape and normalise grayscale image with 255.0
        (iii) one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]
        """
        data['pixels'] = data['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
        data_X = np.array(data['pixels'].tolist(), dtype='float32').reshape(-1,self.width, self.height,1)/255.0   
        data_Y = to_categorical(data['emotion'], self.num_classes)  
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
        #self.__model = model
        
        #model.compile(optimizer= 'adam' , loss= keras.losses.binary_crossentropy, metrics=['accuracy'])
        model.compile(optimizer= 'adam' , loss= 'categorical_crossentropy', metrics=['accuracy'])
        #model.summry()
        model.summary()
        
        
        # train the network
        print("[INFO] Training the network...")
        print()
        
        
        #train the model by slicing the data into "batches" of size batch_size, 
        #and repeatedly iterating over the entire dataset for a given number of epochs.
        
        
        history = model.fit_generator(data_generator.flow(train_X, train_Y, self.batch_size),
                                      steps_per_epoch=len(train_X) / self.batch_size,
                                      epochs=self.num_epochs,
                                      verbose=2, 
                                      callbacks = [es],
                                      validation_data=(val_X, val_Y))
        
        #history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=300, verbose=0)
        
        # Evaluate the model on the test data using `evaluate`
        print('Evaluate on test data')
        
        results = model.evaluate(test_X, test_Y, batch_size=32)
        
        print('test loss ' + str(results[0])  + ' , test acc ' + str(results[1])) 
        
        
        
        
        return history
    
   
    def __plot_lossAndacc(self, history):
        """
        history: holds a record of the loss values and metric values during training
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
        
        
        """
    def predictRandomImage(self):
        
        Generate predictions (probabilities -- the output of the last layer)
        on new data using `predict`
        
        test_X, test_Y   = self.__CRNO(self.data_test, "test") #test data
        
        #model = my_Model.my_model.build_model(self.width,self.height,self.num_classes,self.num_features)

        #num = random.randint()
        print('Predict the classes: ')
        prediction = self.model.predict(test_X[0:1])
        print('Predicted class: ', prediction)
        print('Real class:  ', test_Y[0:1])
        
    def predictRandomImage2(self):
        pixels_colmn = self.data.iloc[:,-2]
        emotion_colmn = self.data.iloc[:,-3]
        
        #(trainX, testX, trainY, testY) = train_test_split(pixels_colmn, emotion_colmn, test_size=0.2, random_state=42)
        test_X, test_Y   = self.__CRNO(self.data_test, "test") #test data
        train_X, train_Y = self.__CRNO(self.data_train, "train") #training data

        
        
        y_pred = self.model.predict(train_X, batch_size=100)
        y_pred1D = y_pred.argmax(1)
        y_test1D = test_Y.argmax(1)
        print ('Accuracy on validation data: ' + str(accuracy_score(y_test1D, y_pred1D)))
        print()
        score_Keras = self.model.evaluate(test_X, test_Y, batch_size=200)
        print('Accuracy on validation data with Keras: ' + str(score_Keras[1]))


    def PredictEmotion(self):
        self.model.compile(optimizer= 'adam' , loss= 'categorical_crossentropy', metrics=['accuracy'])

        train_X, train_Y = self.__CRNO(self.data_train, "train") #training data
        test_X, test_Y   = self.__CRNO(self.data_test, "test") #test data
        _, train_acc = self.model.evaluate(train_X, train_Y, verbose=0)
        _, test_acc = self.model.evaluate(test_X, test_Y, verbose=0)
        print()
        print('Train: {}, Test: {}'.format(train_acc, test_acc))



        



emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer      
data = unload_dataset.open_dataset(file_path)
data = unload_dataset.fixUsageValues(data, emotion_map)

data_train = data[data['Usage']=='Training'].copy()
data_val   = data[data['Usage']=='PublicTest'].copy()
data_test  = data[data['Usage']=='PrivateTest'].copy()
tr = training_Model(data, data_train, data_val, data_test)
tr.PredictEmotion()
"""
    
        
