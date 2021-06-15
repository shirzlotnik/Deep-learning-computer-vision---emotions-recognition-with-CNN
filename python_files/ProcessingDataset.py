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
        

        
        
    def __Setup_axe(self,usage, title):
        """
        title: graph title- str
        sort the dataset by emotion and plot emotion number bar graph
        """
        li = [self.data_train, self.data_val, self.data_test]
        occ_data = li[usage]
        emotion_counts = occ_data['emotion'].value_counts(sort=True).reset_index()
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
        the function sort the emotion column in data by usage 
        return: train_sorted, val_sorted, test_sorted- sorted data by usage
        """
        
        train1 = self.data_train['emotion'].value_counts().sort_index()
        val1 = self.data_val['emotion'].value_counts().sort_index()
        test1 = self.data_test['emotion'].value_counts().sort_index()
        
        train_sorted = sorted(train1.items(), key = lambda d: d[1], reverse = True)
        val_sorted = sorted(val1.items(), key = lambda d: d[1], reverse = True)
        test_sorted = sorted(test1.items(), key = lambda d: d[1], reverse = True)
        
        
        return train_sorted, val_sorted, test_sorted
    

    def __print_emotions_for_usage(self, columns_count, data_sorted, usage):
        """
        columns_count: data_train.shape[0]- how many of that usage total
        data_sorted: the sorted data by usage and then by emotion from 
            count_emotion_in_columns()
        usage: string, the usage
        print number of *emotion* in *usage*
        """
        for info in data_sorted:
            print('Number of {} in {} = {} => {:.4f}%'.format(
                    self.emotion_map.get(info[0]),
                  usage, info[1], (info[1]/columns_count)*100))
        
    def plot_emotions_forUsage(self):
        """
        plot bar graph for each of the usages -> number of images per emotion
        print precentage of emotion from total images in usage
        """
        (trainSort, valSort, testSort) = self.__sort_by_usage()
        dataFexpress_tuple = (trainSort, valSort, testSort)
        dataUsage_tuple = (self.data_train, self.data_val, self.data_test)
        dictFor_tuples = {0:['train', 'Training data'], 1:['validation',
                          'Validation data'], 2:['test','Testing data']}

        for info in range(3):
            self.__Setup_axe(info, dictFor_tuples.get(info)[1])
            print('Emotion Distribution in {}\n'.format(
                    dictFor_tuples.get(info)[1]))
            self. __print_emotions_for_usage(dataUsage_tuple[info].shape[0], 
                        dataFexpress_tuple[info], dictFor_tuples.get(info)[1])
            print()
        print()
            
            
                
