#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:44:33 2021

@author: shirzlotnik
"""

#Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import random
import os
#libaries for the model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt #for the graphs and images
import seaborn as sns #for the graphs


print()
file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer

data = pd.read_csv(file_path)  #open dataset with pandas

"""
data.loc[-1] = [7,None,'Training'] # add 'other' column in case the image is not any of the emotions
data.loc[-2] = [7,None,'PublicTest']
data.loc[-3] = [7,None,'PrivateTest']
"""
def check_data_(data):
    """
    data: the dataset- DataFrame object
    the function print to user the shape of the dataset, the first 5 lines 
    and the usage values suppose to be 80% training, 10% validation and 10% test
    """
    print(data.shape) # check data shape
    print()
    print(data.head(5)) # preview first 5 row of data
    print()
    print(data.Usage.value_counts()) # check usage values
    print()
    
emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral', 7: 'Other'}

def check_target_labels(data, emotion_map):
    """
    data: the dataset- DataFrame type
    emotion_map: dictionery for the diffrent facial expression
    the function sort and print for the user the number of images for each facial expression
    return: emotion_counts- DataFrame that show how many images for each facial expression
    """
    emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
    emotion_counts.columns = ['emotion', 'number']
    emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
    print()
    return emotion_counts

emotion_counts = check_target_labels(data,emotion_map)

def Print_enotion_Counts():
    print(emotion_counts)

def plot_class_distribution(emotion_counts):
    """
    emotion_counts: DataFrame that show how many images for each facial expression
    the function using matplotlib libary to plot to the user a bar graph for the emotion_counts
    """
    plt.figure(figsize=(6,4))
    sns.barplot(emotion_counts.emotion, emotion_counts.number)
    plt.title('Class distribution')
    plt.ylabel('Number', fontsize=12)
    plt.xlabel('Emotions', fontsize=12)
    plt.show()

# plot some images

def row2image(row):
    '''
    row: row from the dataset, type='pandas.core.series.Series'
    the function takes the information from the pixels and emotion columns and tranfer it to 48*48 image
    return: 'numpy.ndarray' of the image
    '''
    pixels, emotion = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split())
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return np.array([image.astype(np.uint8), emotion])

def plot_images(data):
    """
    data: the dataset- DataFrame object
    the function uses matplotlib libary to plot the images to the user
    """
    plt.figure(0, figsize=(12,6))
    for i in range(1,8):
        face = data[data['emotion'] == i-1].iloc[0]
        img = row2image(face)
        plt.subplot(2,4,i)
        plt.imshow(img[0])
        plt.title(img[1])
    plt.show()
