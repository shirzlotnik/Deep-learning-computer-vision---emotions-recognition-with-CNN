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

import PrintsForUser

#file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer
#zip_path = '/Users/shirzlotnik/emotion_dataset.zip'
#emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}



def open_dataset(file_path):
    """
    ask the user to put file path for dataset or zip or use deafult path
    """
    data = None
    li = file_path.split('/')
    if li[len(li)-1] == 'emotion_dataset.zip':
        zf = zipfile.ZipFile(zip_path)
        data = pd.read_csv(zf.open('fer2013.csv')) #fer2013.csv - file name
    elif li[len(li)-1] == 'fer2013.csv':
        data = pd.read_csv(file_path)
    else:
        data = None
        return data
    return data



def fixUsageValues(data, emotion_map):
    """
    data: the dataset- DataFrame object
    emotion_counts:
    transer images randomly from training to validation so the dataset will be splited 
    70% training, 20% validation and 10% test
    """
    # find how many images we need to tranfet for each emotion -> proportions
    emotion_counts = check_target_labels(data, emotion_map)
    number_list = []
    for info in range(len(emotion_counts.values)):
        number_list.append(emotion_counts.values[info][1])
    
    transferImg_list = []
    for info in range(len(number_list)):
        transferImg_list.append(int(number_list[info] / 10)) # 10% of total number of images per expewssion

    
    # change 10% of each facial expression from training to validation
    for i in range(7):
        facex_data = data.nsmallest(number_list[i], 'emotion', keep='first')
        indexList = getIndexList(facex_data)
        for a in range(transferImg_list[i]):
            data.ix[indexList[a], 'Usage'] = 'PublicTest'
            
    return data

        
    
def getIndexList(facex_data):
    """
    facex_data: slip pandas series object by facial expression
    return: list of indexes with same facial expression
    """
    dict_data = facex_data.to_dict('split')
    index_list = dict_data.get('index')
    return index_list



def check_data(data):
    """
    data: the dataset- DataFrame object
    the function print to user the shape of the dataset, the first 5 lines 
    and the usage values suppose to be 80% training, 10% validation and 10% test
    """
    print('Dataset Information')
    print(data.shape) # check data shape
    print()
    print(data.head(5)) # preview first 5 row of data
    print()
    print(data.Usage.value_counts()) # check usage values
    

def check_target_labels(data, emotion_map):
    """
    data: the dataset- DataFrame type
    emotion_map: dictionery for the diffrent emotions 
    the function sort and print for the user the number of images for each facial expression
    return: emotion_counts- DataFrame that show how many images for each facial expression
    """
    emotion_counts = data['emotion'].value_counts(sort=True).reset_index()
    emotion_counts.columns = ['emotion', 'number']
    emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
    print()
    return emotion_counts

#facial_counts = check_target_labels(data, expression_map)
#new_data = fixUsageValues(data, facial_counts)

    


def plot_class_distribution(emotion_counts):
    """
    emotion_counts: DataFrame that show how many images for each emotion
    the function using matplotlib libary to plot to the user a bar graph for the facial_counts
    """
    plt.figure(figsize=(6,4))
    sns.barplot(emotion_counts.emotion, emotion_counts.number)
    plt.title('Class Distribution')
    plt.ylabel('Number', fontsize=12)
    plt.xlabel('Emotions', fontsize=12)
    plt.show()

# plot some images

def row2image(row, emotion_map):
    '''
    row: row from the dataset, type='pandas.core.series.Series'
    emotion_map: 
    the function takes the information from the pixels and emotion columns and tranfer it to 48*48 image
    return: 'numpy.ndarray' of the image
    '''
    pixels, emotions = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split())
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return np.array([image.astype(np.uint8), emotions])

def plot_images(data, emotion_map):
    """
    data: the dataset- DataFrame object
    emotion_map:
    the function uses matplotlib libary to plot the images to the user
    """
    plt.figure(0, figsize=(12,6))
    for i in range(1,8):
        face = data[data['emotion'] == i-1].iloc[0]
        img = row2image(face, emotion_map)
        plt.subplot(2,4,i)
        plt.imshow(img[0])
        plt.title(img[1])
    plt.show()
    
    
def handle_unloadDataset(emotion_map, file_path):
    """
    a method that sums up all the methods together
    """
    li = file_path.split('/')
    if li[len(li)-1] != 'emotion_dataset.zip' and li[len(li)-1] != 'fer2013.csv':
        PrintsForUser.print_error('[ERROR] path not valid plaese check file_path value')
        return
    else:
        data = open_dataset(file_path)
    data = fixUsageValues(data, emotion_map) #fix train, val, tets values
    emotion_counts = check_target_labels(data, emotion_map)
    check_data(data)
    plot_class_distribution(emotion_counts)
    plot_images(data, emotion_map)

    
    
    
 
