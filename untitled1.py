#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:05:20 2018

@author: kevin
"""

N = shape(errors)
mini = errors[0,0]
index =[]
for i in range(N[0]):
    for j in range(N[1]):
        if errors[i,j]<= mini:
            mini = errors[i,j]
            index=[i,j]
            
opt_a = C_range[index]
print('alpha value %d has lowest error of %d' %(opt_a, mini))