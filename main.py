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

#PM2.5 for training
#PM25_train = y_train
errors=[]
i = 0;
mse = 0;
lists = [ 0, 1e-5, 1e-8, 0.1, 1, 10, 100, 1000] 
for a in lists:
    #Do ridge regression return coefficients and error 
    #coef, mse_train = ridge(X_train_scaled, y_train, a)
    #Cross validation splitter
    temp = mse
    mse = cross_valid(X_train_scaled,y_train, a)
    #mse = np.asscalar(mse)
    mse = float(mse[0])
    if mse <= temp:
        mini = mse
        index = i
    i += 1    
    errors.append(mse)
    
val = lists[index]
 
print('alpha value %d has lowest error of %d' %(val, mini))
del errors, i, val, mini 


    



# sdel  X_train, X_test, y_train
