#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:11:24 2018

@author: kevin
"""

import numpy as np

from sklearn import preprocessing
import matplotlib.pyplot as plt
from data2matrix import data2matrix
from cross_valid import cross_valid
from normalise import normalise
from set_split import split
from test_error import testing


#import data from csv file to a matirx
data = data2matrix()

X_train, X_test, y_train, y_test = split(data)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
#nomalise test set with the mean and std from training set
X_train_scaled, X_test_scaled = normalise(X_train, X_test)
#to exam
#new_mean = train_scaled.mean()
#new_std = train_scaled.std()

#PM2.5 for training
#PM25_train = y_train
errors=[]
i = 0;
index = 0;
mse = 0;

#alphas = [ 1e-6, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
alphas = np.logspace(-8, 8, 250)
for a in alphas:
    #Cross validation splitter
    mse = cross_valid(X_train_scaled, y_train, 'Lasso', a, 0.5)
    #mse = float(mse[0]) 
    errors.append(mse)
    
#search for loweest error and correspondsing alpha
mini = errors[0]
for i in range(len(errors)):
    if errors[i] <= mini:
        mini = errors[i]
        index = i
    i += 1 
    
opt_a = alphas[index]
 
print('alpha value %d has lowest error of %d' %(opt_a, mini))
#del  i,a, val, mini, mse, temp

#Plot error against alpha
plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('mean squared error')
plt.title('CV error versus the regularization')
plt.axis('tight')
plt.show()
'''

test_e = testing(X_train_scaled,y_train,X_test_scaled,y_test,'Lasso',1, 0.5, 'linear')

print('testing error is %d' %test_e)
 


del i, index, mse
'''