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
    X = data[:,[0,1,2,4,5,6,7,8,9,10]]
    #PM25 is on column 4
    y = data[:,3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    split(data)