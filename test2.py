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

X_train_scaled, X_test_scaled = normalise(X_train, X_test)
X_train, X_test, y_train, y_test = split(data)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


y = data[:,3]
