#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:16:31 2018

@author: kevin
"""
import numpy as np
import pandas as pd

def data2matrix():
    read = pd.read_csv('PRSA_data.csv').as_matrix()
    for i in range (43824):
        temp = read[i,9]
        if temp == 'NE':
            read[i,9] = 1
        elif temp == 'SE':
            read[i,9] = 2
        elif temp == 'NW':
            read[i,9] = 4
        elif temp == 'cv':
            read[i,9] = 0
        del temp, i
    #df = pd.DataFrame(read)
    data = np.matrix(read) 
    return data

if __name__ == '__main__':
    data2matrix()
    

#csv = np. genfromtxt('PRSA_data.csv',dtype=float, delimiter=",")






    