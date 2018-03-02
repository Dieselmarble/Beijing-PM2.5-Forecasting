#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:50:46 2018

@author: kevin
"""

import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from ridge_regression import ridge
from data2matrix import data2matrix
from set_split import split
from normalise import normalise


data = data2matrix()

X_train, X_test, y_train, y_test = split(data)
#nomalise test set with the mean and std from training set
X_train_scaled, X_test_scaled = normalise(X_train, X_test)
#Return the number of splitting iterations in the cross-validator 
#kf.get_n_splits(train)
fold = 0
X_train1 = []
X_test1 = []
y_train1 = []
y_test1 = []
e = 0;

#number of folds 
#10 folds, each group has 3505 rows of test data(refleted in test_idx)
#(K-1)N/K = 31554 training points 
kf = KFold(n_splits = 10, shuffle = False, random_state=None)
y = y_train

for train_idx, test_idx in kf.split(X_train_scaled):
    fold += 1
    
    X_train1 = X_train_scaled[train_idx]
    y_train1 = y[train_idx]
    X_test1 = X_train_scaled[test_idx]
    y_test1 = y[test_idx]
    coef, mse_train = ridge(X_train1, y_train1)
    y_predict = np.dot(X_test1,np.transpose(coef))
    error = y_test1 - y_predict
    error = np.resize(error,shape)
    e = e + error
    
mse = np.dot(np.transpose(e),e)/len(X_test)

#del fold



