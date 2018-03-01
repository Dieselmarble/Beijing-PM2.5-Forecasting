#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:31:15 2018

@author: kevin
"""

from sklearn import preprocessing
import numpy as np
import math


def normalise(train, test):
    mean = train.mean(axis = 0)
    std =  train.std(axis = 0)
    std = remove_zero(std)
    test_scaled = (test - mean)/std
    train_scaled = (train - mean)/std
    return train_scaled, test_scaled

def remove_zero(array):
    N = np.shape(array)
    nu_cols = N[1]
    for i in range (0,nu_cols):
        if array[0,i]==0:
            array[0,i]=1
    return array

if __name__ == '__main__':
    normalise(train, test)
    

    
