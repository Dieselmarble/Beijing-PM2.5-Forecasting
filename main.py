#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:11:24 2018

@author: kevin
"""

import numpy as np
from sklearn.model_selection import train_test_split
#from data2matrix import data
from data2matrix import data2matrix
from cross_valid import cross_valid

#import data from csv file to a matirx
data = data2matrix()
#split data into training and test set
train, test = train_test_split(data, test_size=0.2)

(X_train,X_valid)=cross_valid(train)

