#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:11:24 2018

@author: kevin
"""

import numpy as np

from sklearn import preprocessing
#from data2matrix import data
from data2matrix import data2matrix
from cross_valid import cross_valid
from normalise import normalise
from ridge_regression import ridge
from set_split import split

#import data from csv file to a matirx
data = data2matrix()

X_train, X_test, y_train, y_test = split(data)

#nomalise test set with the mean and std from training set
X_train_scaled, X_test_scaled = normalise(X_train, X_test)
#to exam
#new_mean = train_scaled.mean()
#new_std = train_scaled.std()

#Cross validation splitter
(X_train_set,X_valid,y_train_set,y_valid) = cross_valid(X_train_scaled)

#PM2.5 for training
PM25_train = y_train

#Do ridge regression return coefficients and error 
coef,mse_train = ridge(X_train_scaled, PM25_train)

del data, X_train, X_test, y_train
