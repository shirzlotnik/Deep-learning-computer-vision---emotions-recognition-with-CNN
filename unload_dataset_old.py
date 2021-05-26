#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:44:33 2021

@author: shirzlotnik
"""

#Import libraries


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile

import matplotlib.pyplot as plt #for the graphs and images
import seaborn as sns #for the graphs

file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer
zip_path = '/Users/shirzlotnik/emotion_dataset.zip'
expression_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}





def open_dataset():
    """
    ask the user to put file path for dataset or zip or use deafult path
    """
    print('Do you want to use deafult path or enter file or zip file?')
    print('Please press f for dataset file')
    print('             z for zip file')
    print('             any other key for deafult')

    ans = input('Enter => ')
    if ans == 'z':
        print('Please enter the path of the zip file in your computer')
        print('Path example for windows -->  C:\\Users\\User\\Desktop\\example.zip')
        print('Path example for macOs -->  /Users/shirzlotnik/emotion_dataset/example.zip')
        zip_file_path = input('Enter zip file path => ')
        zf = zipfile.ZipFile(zip_file_path)
        data = pd.read_csv(zf.open('fer2013.csv')) #fer2013.csv - file name
    if ans == 'f':
        print('Please enter the path of the dataset file in your computer')
        print('Path example for windows -->  C:\\Users\\User\\Desktop\\example.csv')
        print('Path example for macOs -->  /Users/shirzlotnik/emotion_dataset/example.csv')
        data_path = input('Enter dataset file path => ')
        data = pd.read_csv(data_path) 
    else:
        data = pd.read_csv(file_path)
    ######
    data = data.rename(columns={'emotion': 'expression'})

    return data

        
        
print()



#data = pd.read_csv(file_path)  #open dataset with pandas


"""
data.loc[-1] = [7,None,'Training'] # add 'other' column in case the image is not any of the emotions
data.loc[-2] = [7,None,'PublicTest']
data.loc[-3] = [7,None,'PrivateTest']
"""
def check_data(data):
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
    

def check_target_labels(data, expression_map):
    """
    data: the dataset- DataFrame type
    expression_map: dictionery for the diffrent facial expression
    the function sort and print for the user the number of images for each facial expression
    return: emotion_counts- DataFrame that show how many images for each facial expression
    """
    facial_counts = data['expression'].value_counts(sort=True).reset_index()
    facial_counts.columns = ['expression', 'number']
    facial_counts['expression'] = facial_counts['expression'].map(expression_map)
    print()
    return facial_counts

#facial_counts = check_target_labels()
#print(facial_counts)


def plot_class_distribution(facial_counts):
    """
    facial_counts: DataFrame that show how many images for each facial expression
    the function using matplotlib libary to plot to the user a bar graph for the facial_counts
    """
    plt.figure(figsize=(6,4))
    sns.barplot(facial_counts.expression, facial_counts.number)
    plt.title('Class Distribution')
    plt.ylabel('Number', fontsize=12)
    plt.xlabel('Expressions', fontsize=12)
    plt.show()

# plot some images

def row2image(row):
    '''
    row: row from the dataset, type='pandas.core.series.Series'
    the function takes the information from the pixels and emotion columns and tranfer it to 48*48 image
    return: 'numpy.ndarray' of the image
    '''
    pixels, express = row['pixels'], expression_map[row['expression']]
    img = np.array(pixels.split())
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return np.array([image.astype(np.uint8), express])

def plot_images(data):
    """
    data: the dataset- DataFrame object
    the function uses matplotlib libary to plot the images to the user
    """
    plt.figure(0, figsize=(12,6))
    for i in range(1,8):
        face = data[data['expression'] == i-1].iloc[0]
        img = row2image(face)
        plt.subplot(2,4,i)
        plt.imshow(img[0])
        plt.title(img[1])
    plt.show()
    
    
def handle_unloadDataset(data, expression_map):
    """
    a method that sums up all the methods together
    """
    check_data(data)
    facial_counts = check_target_labels(data, expression_map)
    plot_class_distribution(facial_counts)
    plot_images(data)
