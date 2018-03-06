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
    #columns 0 - 80
    X = data[:, 0:54]
    #PM25 to be predicted is on column 3
    y = data[:,54]
    X_train, X_test, y_train, y_test = train_test_split\
    (X, y, test_size=0.15,shuffle=False) 
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    split(data)