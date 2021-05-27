#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:51:59 2021

@author: shirzlotnik
"""


"""
this is the main python file that manege the program according to the user choicess
"""

import unload_dataset
import ModelTrain
import ProcessingDataset

 



file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer
expression_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


data = unload_dataset.open_dataset()
# update dataset usage to 70% training, 20% validation and 10% test

facial_counts = unload_dataset.check_target_labels(data, expression_map)
new_data = unload_dataset.fixUsageValues(data, facial_counts)
data = new_data

data_train = data[data['Usage']=='Training'].copy()
data_val   = data[data['Usage']=='PublicTest'].copy()
data_test  = data[data['Usage']=='PrivateTest'].copy()

#emotion_counts = unload_dataset.emotion_counts


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

def case_one(data):
    """
    data: the dataset- DataFrame type
    the function calls module unload_dataset to download the dataset and open it with pandas
    and plot some information about the dataset
    """
    """
    unload_dataset.extractDatasetFromZip()
    
    unload_dataset.check_data_()
    #unload_dataset.Print_enotion_Counts()
    unload_dataset.check_target_labels()
    unload_dataset.plot_class_distribution()
    unload_dataset.plot_images()
    """
    unload_dataset.handle_unloadDataset(data, expression_map)

def case_two(data, expression_map): 
    """
    the function calls module preprocessing to process the dataset and present to the user with graph
    and numbers
    """
    ######
    
    """
    preprocessing.print_data_usage_info()
    print()
    preprocessing.build_graphs_numberOfExp_forUsage()
    """
    
    process_obj = ProcessingDataset.ProcessDataset(data, expression_map, data_train, data_val, data_test)
    process_obj.handel_process_dataset()
    
    

def case_three(data):
    """
    the model is built and trained, and the weights will be saved
    the visualize training performance will be as well in this part, additional graphs and
    analysis will be added later on
    """
    """
    #initilize parameters
    num_classes = 7 
    width, height = 48, 48
    #num_epochs = 50
    batch_size = 64
    num_features = 64
    num_epochs = input('Please enter numer of epoch => ')
    model = my_Model.my_model.build_model(width, height, num_classes, num_features)
    """
    """
    data_train = data[data['Usage']=='Training'].copy()
    data_val   = data[data['Usage']=='PublicTest'].copy()
    data_test  = data[data['Usage']=='PrivateTest'].copy()
    """
    train_obj = ModelTrain.training_Model(data, data_train, data_val, data_test)
    train_obj.handle_train()
    
    
    

def main_menu():
    flag = True
    user_options()
    """
    defult directories
    this directoties change according to the user activity
    """
    #directories_file.create_directories_file()
    
    while(flag):
        #print("--> Your Choice: ")
        choice = input("==> Enter Your Choice: ")
        print()
        
        if choice == '1':
            """
            if the use enter 1 -> the datasat download
            """
            case_one(data) 
            print()
            print("[INFO] Download dataset succecfuly \n")
            #print()
    
        if choice == '2':
            """
            if the use enter 2 -> the pre-processing of the data begins, the user will see how the data is
            sorted, how many images there are with graph.
            """
            print("[INFO] Process of the dataset")
            case_two(data, expression_map)
            
         
        if choice == '3':
            """
            if the use enter 3 -> the program will first call the module my_Model there the model is built
            after the model is finishied we call it from the trainig module
            """
            case_three(data)
            
        if choice == '4':
            case_Four(model_path,labels_path, data_set, new_images_folder)
            
        if choice == ' ':
            print("[INFO] Exiting...")
            flag = False
    
    
if __name__ == "__main__":
    main_menu()



