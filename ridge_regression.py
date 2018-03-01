#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:26:27 2018

@author: kevin
"""

from sklearn.linear_model import Ridge
import numpy as np

def ridge(x,y):        
    clf = Ridge(alpha=1.0)
    clf.fit(x, y)
    coefs=[]
    coefs.append(clf.coef_)
    predict = clf.predict(x)
    e = predict - y

    total_error = np.dot(np.transpose(e),e)
    mse_train = total_error/len(predict)
    return coefs, mse_train

if __name__ == '__main__':
    ridge(x, y)    

    

    
