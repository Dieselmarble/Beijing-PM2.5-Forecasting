#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 22:20:47 2018

@author: kevin
"""
from sklearn.linear_model import ElasticNet
import numpy as np

def elastic(x,y,a,l1):  

    clf = ElasticNet(alpha = a, l1_ratio = l1 ,random_state=0)
    clf.fit(x, y)
    coefs = clf.coef_

    return coefs

if __name__ == '__main__':
    elastic(x, y, a, l1)    

