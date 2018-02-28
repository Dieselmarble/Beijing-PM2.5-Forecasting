#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:11:24 2018

@author: kevin
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from data2matrix import data
from data2matrix import data2matrix
from cross_valid import cross_valid

#import data from csv file to a matirx
data = data2matrix()
#split data into training and test set

train, test = train_test_split(data, test_size=0.2)
#normalise training set
train_scaled = preprocessing.scale(train)
#nomalise test set with the way
#(testData - mean(trainData)) / sd(trainData)

test_scaled = (test-train_scaled.mean(axis=0))/

#X_scaled.std(axis=0)


(X_train,X_valid)=cross_valid(train)

