#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:01:48 2018

@author: kevin
"""
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

def testing(X_train,y_train,X_test,y_test,name,a,l1,kernel_n):
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
    #y_predict = np.ravel(y_predict)
    error = mean_squared_error(y_test, y_predict)
    return error

if __name__ == '__main__':
    testing(X_train,y_train,X_test,y_test,name,a,l1,kernel_n )