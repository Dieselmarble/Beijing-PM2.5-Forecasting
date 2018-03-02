#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:06:00 2018

@author: kevin
"""
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from ridge_regression import ridge

def cross_valid(feature, y, a):

    #number of folds 
    #10 folds, each group has 3505 rows of test data(refleted in test_idx)
    #(K-1)N/K = 31554 training points 
    kf = KFold(n_splits = 10, shuffle = True, random_state=None)
    #Return the number of splitting iterations in the cross-validator 
    #kf.get_n_splits(train)
    fold = 0
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    e = 0;
    for train_idx, test_idx in kf.split(feature):
        
        fold += 1
        X_train = feature[train_idx]
        y_train = y[train_idx]
        X_test = feature[test_idx]
        y_test = y[test_idx]
        coef, mse_train = ridge(X_train, y_train, a)
        y_predict = np.dot(X_test,np.transpose(coef))
        #error of each validation set
        error = y_test - y_predict
        #accumalated error
        error = np.resize(error,(3341,1))
        e = e + error
           
    #Calculate Mean Squared Error    
    mse = np.dot(np.transpose(e),e)/len(X_test)
    #Then divided by number of folds getting Cross-validation error 
    mse = mse/fold
    del fold
    return mse
    
if __name__ == '__main__':
    cross_valid(feature, y, a)
    
