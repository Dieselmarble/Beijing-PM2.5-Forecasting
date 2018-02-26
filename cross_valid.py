#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:06:00 2018

@author: kevin
"""
import numpy as np

from sklearn.model_selection import KFold
def cross_valid(train):
    X = train
    n = len(X)
    #number of folds 
    #10 folds, each group has 3505 rows of test data(refleted in test_idx)
    #(K-1)N/K = 31554 training points 
    kf = KFold(n_splits = 10, shuffle = True, random_state=None)
    #Return the number of splitting iterations in the cross-validator 
    #kf.get_n_splits(train)
    fold = 0
    X_train = []
    X_test = []

    
    for train_idx, test_idx in kf.split(train):
        fold += 1
        X_train.append(train_idx)
        X_test.append(test_idx)
        
        #train your data
        #clf = LogisticRegression().fit(X_train, X_train(test))
        #test your data
        #score = clf.score(X_test, Y_test)
        #print("score for fold %d: %.3f" %(fold, score))
        #return score
    return X_train, X_test
if __name__ == '__main__':
    cross_valid(train)
    
