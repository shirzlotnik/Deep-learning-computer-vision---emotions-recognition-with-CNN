#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:51:59 2021

@author: shirzlotnik
"""


"""
this is the main python file that manege the program according to the user choicess
"""

import unload_dataset_ForWin
import ModelTrain
import ProcessingDataset
import PrintsForUser







def user_options():
    """
    This function prints for the user his options (UI)
    """
  
    PrintsForUser.print_option("* * * * * * * * * * * * * * * * * * * * * *")
    PrintsForUser.print_option("*              USER INTERFACE             *")
    PrintsForUser.print_option("*                                         *")
    PrintsForUser.print_option("*   Enter 1 -->  download the dataset     *")
    PrintsForUser.print_option("*   Enter 2 --> process dataset           *")
    PrintsForUser.print_option("*   Enter 3 --> train the model and       *")
    PrintsForUser.print_option("*           Evaluate Test Performance     *")
    PrintsForUser.print_option("*   Enter space bar --> exit              *")
    PrintsForUser.print_option("*                                         *")
    PrintsForUser.print_option("* * * * * * * * * * * * * * * * * * * * * *")
#user_options()

def case_one(file_path, emotion_map):
    """
    file_path: path to the place the dataset or ziped dataset are
    the function calls module unload_dataset to download the dataset and open it with pandas
    and plot some information about the dataset
    """
    #data , fl = unload_dataset.open_dataset(file_path)
    data = unload_dataset.handle_unloadDataset(emotion_map, file_path)
    return data

def case_two(data, emotion_map): 
    """
    the function calls module preprocessing to process the dataset and present to the user with graph
    and numbers
    """
    ######
    
    data_train = data[data['Usage']=='Training'].copy()
    data_val   = data[data['Usage']=='PublicTest'].copy()
    data_test  = data[data['Usage']=='PrivateTest'].copy()
    
    process_obj = ProcessingDataset.ProcessDataset(data, emotion_map, data_train, data_val, data_test)
    process_obj.plot_emotions_forUsage()
    
    

def case_three(data):
    """
    the model is built and trained, and the weights will be saved
    the visualize training performance will be as well in this part, additional graphs and
    analysis will be added later on
    """
    
    data_train = data[data['Usage']=='Training'].copy()
    data_val   = data[data['Usage']=='PublicTest'].copy()
    data_test  = data[data['Usage']=='PrivateTest'].copy()
    
    train_obj = ModelTrain.training_Model(data, data_train, data_val, data_test)
    train_obj.handle_train()
    

 
def check_emptyData(data):
    """
    data: DataFrame- the dataset
    checks if the dataset is empty
    return: True if empty
            False if not empty
    """
    try:
        if data.empty:
            PrintsForUser.print_error('[ERROR] Cannot process dataset before dataset was install ')
            PrintsForUser.print_error('        Please press 1 to unload dataset before pressing 2 \n\n')
            PrintsForUser.print_process('[INFO] You need to run the program again\n\n')
            return True
    except AttributeError:
        PrintsForUser.print_error('[ERROR] No dataset has been found ')
        return True

    return False

    
    

def main_menu():
    flag = True
    """
    defult directories - this directoties change according to the user activity
    """
    
    #file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer
    #file_path = r'C:\Users\User\Desktop\fer2013.csv' # file path format for user in windows
    file_path = r'C:\\Users\\Student\\Desktop\\fer2013.csv'
    emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    data = None
    
    while(flag):
        user_options()
        choice = input("==> Enter Your Choice: ")
        print()

        if choice == '1':
            """
            if the use enter 1 -> the datasat download
            """
            PrintsForUser.print_process("[INFO] Downloading dataset ")
            data = case_one(file_path, emotion_map) 
            PrintsForUser.print_process("[INFO] Download dataset succecfuly \n")
    
        if choice == '2':
            """
            if the use enter 2 -> the pre-processing of the data begins, the user will see how the data is
            sorted, how many images there are with graph.
            """ 
            if check_emptyData(data):
                return
            PrintsForUser.print_process("[INFO] Process of the dataset\n")
            case_two(data, emotion_map) 
         
        if choice == '3':
            """
            if the use enter 3 -> the program will first call the module my_Model there the model is built
            after the model is finishied we call it from the trainig module
            """
            case_three(data)
            
        if choice == ' ':
            print("[INFO] Exiting...")
            flag = False
    
    
if __name__ == "__main__":
    main_menu()

