#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:48:33 2018

@author: kevin
"""

from sklearn.model_selection import train_test_split
import numpy as np

def split(data):
    #split data into training and test set
    #have deleted year column
    #X = data[:,[0,1,2,4,5,6,7,8,9,10,11]]
    #PM25 to be predicted is on column 3
    #y = data[:,3]
    i = 16
    X = data[16,:]
    for i in range(len(data)):
        if data[i,2] == 8:
            for j in range(9,17): 
                day_p = data[i-j,:]
                X = np.concatenate((X,day_p),axis=0)
        y = data[i,3]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    split(data)