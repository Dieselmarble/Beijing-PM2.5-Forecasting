#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:11:24 2018

@author: kevin
"""
import numpy as np
import matplotlib.pyplot as plt
from data2matrix import data2matrix
from cross_valid import cross_valid
from normalise import normalise
from set_split import split
from test_error import testing

#import data from csv file to a matirx
data = data2matrix()
X_train, X_test, y_train, y_test = split(data)

#y_train = y_train - np.mean(y_train)
#y_test = y_test - np.mean(y_train)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
#nomalise test set with the mean and std from training set
X_train_scaled, X_test_scaled = normalise(X_train, X_test)
errors = []
coefs = []
errors2 = []
coefs2 = []
mse = 0;
'''
#alphas = [ 1e-6, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
alphas = np.logspace(-4, 7, 100)
for a in alphas:
    #Cross validation splitter
    mse, coef = cross_valid\
    (X_train_scaled, y_train, 'Ridge', a, 0.5,5,10)
    #mse = float(mse[0]) 
    errors.append(mse)
    coefs.append(coef)
#search for loweest error and correspondsing alpha
mini = errors[0]
for i in range(len(errors)):
    if errors[i] <= mini:
        mini = errors[i]
        index = i
    i += 1 
    
opt_a = alphas[index]
 
print('alpha value %d has lowest error of %d' %(opt_a, mini))
'''
C_range = np.logspace(-1, 10, 3)
gamma_range = np.logspace(-1, 10, 3)
e=[]
for C in C_range:
    for gamma in gamma_range:
        #cross_valid(feature, y, name, a, l1, C, gamma)
        mse = cross_valid \
        (X_train_scaled, y_train, 'SVM', 1, 0.5, C, gamma)
        e.append((C, gamma, mse)) 
mini = e[0][2]
for i in range(len(e)):
    if e[i][2]<=mini:
        mini = e[i][2]
        index = i
opt_c=e[i][0]
opt_ga=e[i][1]
print('C value %d, gamma value %d has lowest error of %d' \
      %(opt_c,opt_ga, mini))
'''
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

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) #reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('coefficients versus the regularization')
plt.axis('tight')
plt.show()
'''
'''
train_e, test_e = testing(X_train_scaled,y_train,X_test_scaled,y_test,\
                          'SVM',opt_a, 0.5, 5,10)

print('training error is %d ;testing error is %d' %(train_e,test_e))
'''
