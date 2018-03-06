#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:06:00 2018

@author: kevin
"""
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def cross_valid(feature, y, name, a, l1):

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
    previous_mse = 0;
    for train_idx, test_idx in kf.split(feature):  
        fold += 1
        X_train = feature[train_idx]
        y_train = y[train_idx]
        X_test = feature[test_idx]
        y_test = y[test_idx]
       
        if name == 'Ridge':
            clf = Ridge(alpha = a)
        if name == 'Lasso':
            clf = Lasso(alpha = a)
        if name =='elastic':
            clf = ElasticNet(alpha = a, l1_ratio = l1 ,random_state=0)
        if name == 'SVM':
            clf = SVR(C=1.0, epsilon=0.2, kernel=kernel_n )
        
        clf.fit(X_train,y_train)
        y_predict = clf.predict(X_test)   
        mse = mean_squared_error(y_test, y_predict)
    
        #error of each validation set
        #error = y_test - y_predict
        #error = np.resize(error,(139,1))
        #error = np.resize(error,(3341,1))
        #sum all errors squared
        #sum_square_error = np.dot(np.transpose(error),error)
        #each validation error(MSE)
        #mse = sum_square_error/len(y_test)
        mse = mse + previous_mse
        previous_mse = mse

    #Then divided by number of folds getting Cross-validation error 
    cv_error = mse/fold

    return cv_error
    
if __name__ == '__main__':
    cross_valid(feature, y, name, a, l1)
    
