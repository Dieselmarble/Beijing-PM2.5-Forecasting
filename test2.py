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
X = np.zeros([1738,108])
y = []
index = 0;
for i in range(32,len(data)):
    if data[i,2] == 8:
        for j in range(24,32):
            #X.append(data[i-j,:])
            X[index,j]
        y.append(data[i,3])
        index += 1 
    i += 1
y = np.asarray(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
