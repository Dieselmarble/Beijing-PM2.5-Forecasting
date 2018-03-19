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

def testing(X_train,y_train,X_test,y_test,name,a,l1,C_,gamma_):
    if name == 'Constant':
        y_predict = round(np.mean(y_train))
        e = y_test-y_predict
        error = np.mean(np.square(e, e))
        return error
    
    if name == 'Ridge':
        #If set to false, no intercept will be used in calculations
        clf = Ridge(alpha = a)
    elif name == 'Lasso':
        #clf = Lasso(alpha = a)
        clf = Lasso(alpha = a,fit_intercept=True,tol=0.01,max_iter=50000)
    elif name =='elastic':
        clf = ElasticNet(alpha = a, l1_ratio = l1)
    elif name == 'SVM':
        clf = SVR(C=C_, epsilon=0.1,gamma=gamma_, kernel='rbf' )
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    y_predict_train = clf.predict(X_train)
    #y_predict = np.ravel(y_predict)
    test_error = mean_squared_error(y_test, y_predict)
    train_error = mean_squared_error(y_train, y_predict_train)
    return train_error, test_error

    
if __name__ == '__main__':
    testing(X_train,y_train,X_test,y_test,name,a,l1,C_,gamma_ )