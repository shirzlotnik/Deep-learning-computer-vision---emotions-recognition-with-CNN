#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:13:28 2021

@author: shirzlotnik
"""

import unload_dataset
import matplotlib.pyplot as plt #for the graphs and images
import seaborn as sns #for the graphs
"""
Pre-processing data
1. Splitting dataset into 3 parts: train, validation, test
2. Convert strings to lists of integers
3. Reshape to 48x48 and normalise grayscale image with 255.0
4. Perform one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]
"""

data = unload_dataset.data
emotion_map = unload_dataset.emotion_map


data_train = data[data['Usage']=='Training'].copy()
data_val   = data[data['Usage']=='PublicTest'].copy()
data_test  = data[data['Usage']=='PrivateTest'].copy()

def print_data_usage_info():
    print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(
        data_train.shape, data_val.shape, data_test.shape))
    print()
    
def Setup_axe(data,title):
    """
    data: the dataset- DataFrame object
    title: graph title- str
    sort the dataset by emotion
    """
    
    emotion_Count = data['emotion'].value_counts(sort=False).reset_index()
    emotion_Count.columns = ['emotion', 'number']
    emotion_Count['emotion'] = emotion_Count['emotion'].map(emotion_map)
    # using seaborn libary to plot graphs
    sns.barplot(emotion_Count.emotion, emotion_Count.number)
    plt.title(title)
    plt.ylabel('Number', fontsize=12)
    plt.xlabel('Emotions', fontsize=12)
    plt.show()
    
def count_emotion_in_columns(data):
    """
    data: the dataset- DataFrame object
    the function sort the data by usage and then by facial expression
    return: train_sorted, val_sorted, test_sorted- sorted data by usage
    """
    
    train1 = data_train['emotion'].value_counts().sort_index()
    val1 = data_val['emotion'].value_counts().sort_index()
    test1 = data_test['emotion'].value_counts().sort_index()
    
    train_sorted = sorted(train1.items(), key = lambda d: d[1], reverse = True)
    val_sorted = sorted(val1.items(), key = lambda d: d[1], reverse = True)
    test_sorted = sorted(test1.items(), key = lambda d: d[1], reverse = True)
    
    
    return train_sorted, val_sorted, test_sorted

def print_Usage_Information(columns_count,data_sorted, usage):
    """
    columns_count: data_train.shape[0]- how many of that usage total
    data_sorted: the sorted data by usage and then by emotion from count_emotion_in_columns()
    usage: string, the usage
    print number of *emotion* in *usage*
    """
    for info in data_sorted:
        print('Number of {} in {} = {} => {}%'.format(emotion_map.get(info[0]),
              usage, info[1], (info[1]/columns_count)*100))
        

trainSort, valSort, testSort = count_emotion_in_columns(data)

def build_graphs_numberOfExp_forUsage():
    """
    liDataSort: tuple of the sorted data
    liDatas: tuple of data by usage
    dicTitles: dicteonry of grapfh titles by index
    emotion_map: emotion_map: dictionery of index and facial expression
    the function plot the graphs and print below the information of the graph
    """
    tuDataSort = (trainSort, valSort, testSort)
    tuDataUsage = (data_train, data_val, data_test)
    dictoneryT = {0: ['train', 'training data'], 1: ['validation','validation data'], 2: ['test','testing data']}
    
    for i in range(3):
        Setup_axe(tuDataUsage[i],dictoneryT.get(i)[0])
        print_Usage_Information(tuDataUsage[i].shape[0],tuDataSort[i],dictoneryT.get(i)[1] )

#build_graphs_numberOfExp_forUsage()
