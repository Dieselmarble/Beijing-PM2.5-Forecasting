#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:48:33 2018

@author: kevin
"""

from sklearn.model_selection import train_test_split
import numpy as np

from data2matrix import data2matrix

data = data2matrix()
#split data into training and test set
#have deleted year column
#X = data[:,[0,1,2,4,5,6,7,8,9,10,11]]
#PM25 to be predicted is on column 3
#y = data[:,3]
i = 16
new = data[16,:]
X = np.matrix
y = np.matrix
for i in range(len(data)-8):
    if data[i,2] == 8:
        for j in range(9,17): 
            day_p = data[i-j,:]
            new = np.concatenate((new,day_p),axis=0)
        X.(new)
        new = data[i+8,:]
        y.append(data[i,3])
        index = index + 1 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
