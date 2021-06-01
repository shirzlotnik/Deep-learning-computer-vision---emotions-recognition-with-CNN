#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:20:03 2021

@author: shirzlotnik
"""

"""
this file is in charge for the prints that the user sees in the terminal.
there are 3 types of prints:
    (1) prints that ask the user to give input - will be print in green
    (2) prints that givr information about the project processing - will be print in blue
    (3) prints about errors - will be print in red
"""
from colorama import init
from colorama import Fore, Back, Style


def print_error(mes):
    """
    mes: the messsage that will be print in the terminal to the user
    the function print the message to the user with the right color- error=red
    """
    init()
    print(Fore.RED + mes + Style.RESET_ALL)
    Style.RESET_ALL
    
def print_option(mes):
    """
    mes: the messsage that will be print in the terminal to the user
    the function print the message to the user with the right color- error=red
    """
    init()
    print(Fore.GREEN + mes + Style.RESET_ALL)
    #Style.RESET_ALL
    
def print_process(mes):
    """
    mes: the messsage that will be print in the terminal to the user
    the function print the message to the user with the right color- error=red
    """
    init()
    print(Fore.BLUE + mes + Style.RESET_ALL)
    Style.RESET_ALL
    
    
    
    
    
    
