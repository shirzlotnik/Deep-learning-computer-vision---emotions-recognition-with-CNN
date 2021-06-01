#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:31:02 2021

@author: shirzlotnik
"""



import matplotlib.pyplot as plt #for the graphs and images
import seaborn as sns #for the graphs
"""
Processing dataset
1. Splitting dataset into 3 parts: train, validation, test
2. Convert strings to lists of integers
3. Reshape to 48x48 and normalise grayscale image with 255.0
4. Perform one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]
"""

class ProcessDataset:
    
    def __init__(self, data, emotion_map, data_train, data_val, data_test):
        
        self.data = data
        self.emotion_map = emotion_map
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        
        
    def __print_data_usage(self):
        """
        
        """
        print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(
                self.data_train.shape, self.data_val.shape, self.data_test.shape))
        print()
        

        
      
    def __print_Fexpression_for_usage(self, columns_count, data_sorted, usage):
        """
        def print_Usage_Information(columns_count,data_sorted, usage): => in preprocessing.py
        
        columns_count: data_train.shape[0]- how many of that usage total
        data_sorted: the sorted data by usage and then by emotion from count_emotion_in_columns()
        usage: string, the usage
        print number of *emotion* in *usage*
        """
        for info in data_sorted:
            print('Number of {} in {} = {} => {:.4f}%'.format(self.emotion_map.get(info[0]),
                  usage, info[1], (info[1]/columns_count)*100))
        
        
        ########
    def __Setup_axe(self, title):
        """
        data: the dataset- DataFrame object
        title: graph title- str
        sort the dataset by emotion
        """
        
        emotion_counts = self.data['emotion'].value_counts(sort=True).reset_index()
        emotion_counts.columns = ['emotion', 'number']
        emotion_counts['emotion'] = emotion_counts['emotion'].map(self.emotion_map)
        # using seaborn libary to plot graphs
        sns.barplot(emotion_counts.emotion, emotion_counts.number)
        plt.title(title, fontsize=14)
        plt.ylabel('Number', fontsize=12)
        plt.xlabel('Emotions', fontsize=12)
        plt.show()
        
        
    def __sort_by_usage(self):
        """
        def count_emotion_in_columns(data): => in preprocessing.py
        
        #data: the dataset- DataFrame object
        the function sort the data by usage and then by facial expression
        return: train_sorted, val_sorted, test_sorted- sorted data by usage
        """
        
        train1 = self.data_train['emotion'].value_counts().sort_index()
        val1 = self.data_val['emotion'].value_counts().sort_index()
        test1 = self.data_test['emotion'].value_counts().sort_index()
        
        train_sorted = sorted(train1.items(), key = lambda d: d[1], reverse = True)
        val_sorted = sorted(val1.items(), key = lambda d: d[1], reverse = True)
        test_sorted = sorted(test1.items(), key = lambda d: d[1], reverse = True)
        
        return train_sorted, val_sorted, test_sorted
        
    def __plot_Fexpression_forUsage(self):
        """
        plot bar graph for each of the usages -> number of images per facial expression
        print 
        """
        (trainSort, valSort, testSort) = self.__sort_by_usage()
        dataFexpress_tuple = (trainSort, valSort, testSort)
        dataUsage_tuple = (self.data_train, self.data_val, self.data_test)
        dictFor_tuples = {0:['train', 'Training data'], 1:['validation','Validation data'], 2:['test','Testing data']}
        
        for info in range(3):
            self.__Setup_axe( dictFor_tuples.get(info)[1])
            print('Emotion Distribution in {}\n'.format(dictFor_tuples.get(info)[1]))
            self. __print_Fexpression_for_usage(dataUsage_tuple[info].shape[0], dataFexpress_tuple[info], 
                                                dictFor_tuples.get(info)[1])
            print()
        print()
        
    def handel_process_dataset(self):
        """
        this public method manage the process section
        """
        self.__print_data_usage()
        self.__plot_Fexpression_forUsage()
                
            
            
                
