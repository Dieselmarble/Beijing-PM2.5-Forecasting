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


def cross_valid(feature, y, name, a, l1, C_, gamma_):
    kf = KFold(n_splits = 5)
    fold = 0
    coef = 0
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    pcv_e = 0;
    pt_e = 0;
    for train_idx, test_idx in kf.split(feature):  
        fold += 1
        X_train = feature[train_idx]
        y_train = y[train_idx]
        X_test = feature[test_idx]
        y_test = y[test_idx]

        if name == 'Ridge':
            clf = Ridge(alpha = a)
            clf.fit(X_train,y_train)
            coef = clf.coef_
        if name == 'Lasso':
            clf = Lasso(alpha = a,fit_intercept=True,tol=0.01,max_iter=50000)
            clf.fit(X_train,y_train)
            coef = clf.coef_
        if name =='elastic':
            clf = ElasticNet(alpha = a, l1_ratio = l1 ,random_state=0)
            clf.fit(X_train,y_train)
            coef = clf.coef_
        if name == 'SVM':
            clf = SVR(C=C_, gamma=gamma_, epsilon=0.1, kernel='rbf' )
            clf.fit(X_train,y_train)
        y_predict1 = clf.predict(X_test)   
        y_predict2 = clf.predict(X_train) 
        t_e = mean_squared_error(y_train, y_predict2)
        t_e = t_e + pt_e
        pt_e = t_e
        cv_e = mean_squared_error(y_test, y_predict1)
        cv_e = cv_e + pcv_e
        pcv_e = cv_e

    #Then divided by number of folds getting Cross-validation error 
    cv_error = cv_e/fold
    t_error = t_e/fold

    return cv_error,t_error
    
if __name__ == '__main__':
    cross_valid(feature, y, name, a, l1, C_, gamma_)
    
