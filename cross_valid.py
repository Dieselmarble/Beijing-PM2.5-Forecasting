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
from lasso_regression import lasso
from elastic_regression import elastic


def cross_valid(feature, y, a, name, l1):

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
    previous_mse = 0;
    for train_idx, test_idx in kf.split(feature):  
        fold += 1
        X_train = feature[train_idx]
        y_train = y[train_idx]
        X_test = feature[test_idx]
        y_test = y[test_idx]
        if name == 'Ridge':
            coef = ridge(X_train, y_train, a)
        if name == 'Lasso':
            coef = lasso(X_train, y_train, a)
        if name == 'Elastic':
            coef = elastic(X_train, y_train, a, l1)
        y_predict = np.dot(X_test,np.transpose(coef))
        #error of each validation set
        error = y_test - y_predict
        #error = np.resize(error,(139,1))
        error = np.resize(error,(3341,1))
        #sum all errors squared
        sum_square_error = np.dot(np.transpose(error),error)
        #each validation error(MSE)
        mse = sum_square_error/len(y_test)
        mse = mse + previous_mse
        previous_mse = mse

    #Then divided by number of folds getting Cross-validation error 
    cv_error = mse/fold

    return cv_error
    
if __name__ == '__main__':
    cross_valid(feature, y, a, name, l1)
    
