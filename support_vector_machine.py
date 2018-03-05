#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:57:41 2018

@author: kevin
"""

from sklearn.svm import SVR
import numpy as np

def SVM():
    clf = SVR(C=1.0, epsilon=0.2,kernel='rbf')
    clf.fit(x, y)
    coefs = clf.coef_

if __name__ == '__main__':
    SVM(x, y, kernel, degree, alpha, name, l1)
    