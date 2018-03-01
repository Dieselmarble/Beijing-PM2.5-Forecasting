#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:06:00 2018

@author: kevin
"""
import numpy as np

from sklearn.model_selection import KFold
def cross_valid(train):

    n = len(train)
    #number of folds 
    #10 folds, each group has 3505 rows of test data(refleted in test_idx)
    #(K-1)N/K = 31554 training points 
    kf = KFold(n_splits = 10, shuffle = False, random_state=None)
    #Return the number of splitting iterations in the cross-validator 
    #kf.get_n_splits(train)
    fold = 0
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for train_idx, test_idx in kf.split(train):
        fold += 1
        X_train.append(train_idx)
        X_test.append(test_idx)
        y_train.append(train_idx)
        y_test.append(test_idx)
        
    del fold, n, 
    return X_train, X_test, y_train, y_test
    
if __name__ == '__main__':
    cross_valid(train)
    
