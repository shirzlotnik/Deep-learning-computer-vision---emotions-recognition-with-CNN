#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:51:59 2021

@author: shirzlotnik
"""


"""
this is the main python file that manege the program according to the user choicess
"""

import os
import random
import unload_dataset
import preprocessing
import my_Model
import training

#file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer

data = unload_dataset.data


def user_options():
    """
    This function prints for the user his options (UI)
    """
    print("******************************************")
    print("*             USER INTERFACE             *")
    print("*                                        *")
    print("*   Enter 1 --> download the dataset     *")
    print("*   Enter 2 --> pre-precess of dataset   *")
    print("*   Enter 3 --> train the model          *")
    print("*   Enter 4 --> predict random images    *")
    print("*   Enter space bar --> exit             *")
    print("*                                        *")
    print("******************************************")
    
#user_options()

def case_one(data, emotion_map, emotion_counts):
    """
    data: the dataset- DataFrame type
    the function calls module unload_dataset to download the dataset and open it with pandas
    and plot some information about the dataset
    """
    unload_dataset.check_data_(data)
    #unload_dataset.Print_enotion_Counts()
    unload_dataset.check_target_labels(data,emotion_map)
    unload_dataset.plot_class_distribution(emotion_counts)
    unload_dataset.plot_images(data)


def case_two(): 
    """
    the function calls module preprocessing to process the dataset and present to the user with graph
    and numbers
    """
    preprocessing.print_data_usage_info()
    print()
    preprocessing.build_graphs_numberOfExp_forUsage()

def case_three(width, height, num_classes, num_features):
    """
    the model is built and trained, and the weights will be saved
    the visualize training performance will be as well in this part, additional graphs and
    analysis will be added later on
    """
    model = my_Model.my_model.build_model(width, height, num_classes, num_features)
    
    
    

def main_menu():
    flag = True
    user_options()
    """
    defult directories
    this directoties change according to the user activity
    """
    #directories_file.create_directories_file()
    
    
    while(flag):
        print("--> Your Choice: ")
        choice = input("Enter: ")
        print()
        
        if choice == '1':
            """
            if the use enter 1 -> the datasat download
            """
            case_one(unload_dataset.data, unload_dataset.emotion_map, unload_dataset.emotion_counts) 
            print("[INFO] Downloading dataset")
            print()
    
        if choice == '2':
            """
            if the use enter 2 -> the pre-processing of the data begins, the user will see how the data is
            sorted, how many images there are with graph.
            """
            case_two()
            print("[INFO] Pre-process of the dataset")
         
        if choice == '3':
            """
            if the use enter 3 -> the program will first call the module my_Model there the model is built
            after the model is finishied we call it from the trainig module
            """
            case_Three(model_path,labels_path)
            
        if choice == '4':
            case_Four(model_path,labels_path, data_set, new_images_folder)
            
        if choice == ' ':
            print("[INFO] Exiting...")
            flag = False
    
    
if __name__ == "__main__":
    main_menu()




