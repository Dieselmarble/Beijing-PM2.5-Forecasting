#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:01:48 2018

@author: kevin
"""
from ridge_regression import ridge
from lasso_regression import lasso
from elastic_regression import elastic
import numpy as np

def testing(X_train,y_train,X_test,y_test,name,a,l1 ):
    if name == 'Ridge':
        coef = ridge(X_train, y_train, opt_a)
    if name == 'Lasso':
        coef = lasso(X_train, y_train, opt_a)
    if name=='elastic':
        coef = elastic(X_train, y_train, opt_a, l1)

    y_predict = np.dot(X_test,np.transpose(coef))
    error = y_test - y_predict
    sum_square_error = np.dot(np.transpose(error),error)
    test_error = sum_square_error/len(error)
    test_error = float(test_error[0])
    return test_error


if __name__ == '__main__':
    testing(X_train,y_train,X_test,y_test,name,a,l1 )