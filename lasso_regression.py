#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:38:24 2018

@author: kevin
"""

from sklearn.linear_model import Lasso
import numpy as np

def lasso(x,y,a):  

    clf = Lasso(alpha = a)
    clf.fit(x, y)
    coefs = clf.coef_
    predict = clf.predict(x)
    e = predict - y
    #total_error = np.dot(np.transpose(e),e)
    #mse_train = total_error/len(predict)
    return coefs, e

if __name__ == '__main__':
    lasso(x, y, a)    


