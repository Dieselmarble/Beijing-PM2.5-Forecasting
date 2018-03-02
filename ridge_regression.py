#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:26:27 2018

@author: kevin
"""

from sklearn.linear_model import Ridge
import numpy as np

def ridge(x,y,a):  

    clf = Ridge(alpha = a)
    clf.fit(x, y)
    
    coefs = clf.coef_
    
    predict = clf.predict(x)
    e = predict - y
    
    #total_error = np.dot(np.transpose(e),e)
    #mse_train = total_error/len(predict)
     
    return coefs, a

if __name__ == '__main__':
    ridge(x, y, a)    

    

    
