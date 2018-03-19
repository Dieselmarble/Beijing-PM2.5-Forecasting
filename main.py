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
from random import shuffle
import seaborn as sns
import pandas as pd
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

#import data from csv file to a matirx
data = data2matrix()
X_train, X_test, y_train, y_test = split(data)

ind_list = [i for i in range(len(X_train))]
shuffle(ind_list)
X_train = X_train[ind_list]
y_train = y_train[ind_list]
ind_list = [i for i in range(len(X_test))]
shuffle(ind_list)
X_test = X_test[ind_list]
y_test = y_test[ind_list]
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
#nomalise test set with the mean and std from training set
X_train_scaled, X_test_scaled = normalise(X_train, X_test)
errors = []
coefs = []
train_e=[]
Model = 'Ridge'
'''
alphas = np.logspace(-4, 4, 100)
for a in alphas:
    #Cross validation splitter
    mse, te, coef = cross_valid\
    (X_train_scaled, y_train, Model, a, 0.5,5,10)
    #mse = float(mse[0]) 
    errors.append(mse)
    train_e.append(te)
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

#Plot error against alpha
ax = plt.gca()
ax.plot(alphas, errors, label = 'cv error')
ax.plot(alphas, train_e, label = 'train error')
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('mean squared error')
plt.legend(loc=4)
plt.axis('tight')
plt.title('Lasso training and CV error versus the regularization')
plt.show()

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

C_range = np.logspace(-3, 3, 3)
gamma_range = np.logspace(-3, 3, 3)
errors=[]
parc=[]
parg=[]
count = 0
for C in C_range:
    for gamma in gamma_range:
        #cross_valid(feature, y, name, a, l1, C, gamma)
        mse, te  = cross_valid \
        (X_train_scaled, y_train, 'SVM', 1, 0.5, C, gamma)
        errors.append(mse)
        parc.append(C)
        parg.append(gamma)
        print('almost done %d' %count)
        count +=1
mini = errors[0]
for i in range(len(errors)):
    if errors[i] <= mini:
        mini = errors[i]
        index = i
opt_c = parc[i]
opt_ga = parg[1]
print('C value %d, gamma value %d has lowest error of %d' \
      %(opt_c,opt_ga, mini))

# Plot the surface

parc = np.asarray(parc)
parg = np.asarray(parg)
errors = np.asarray(errors)
errors = np.random.rand(4,4)
plt.imshow(errors, interpolation='nearest', cmap=plt.cm.hot)
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.show()
'''
plotx,ploty, = np.meshgrid(np.linspace(np.min(parc),np.max(parg),10),\
                           np.linspace(np.min(parg),np.max(parc),10))
plotz = interp.griddata((parc,parg),errors,(plotx,ploty),method='linear')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')
fig.colorbar(ax, shrink=0.5, aspect=5)

sns.set()
df = pd.DataFrame.from_dict(np.array([parc,parg,errors]).T)
df.columns = ['X_value','Y_value','Z_value']
df['Z_value'] = pd.to_numeric(df['Z_value'])
pivotted= df.pivot('Y_value','X_value','Z_value')
ax = sns.heatmap(pivotted,cmap='RdBu')
ax.set_title('CV errror over C and $\gamma$')
ax.set_xlabel('C')
ax.set_ylabel('$\gamma$')
'''
'''
train_e, test_e = testing(X_train_scaled,y_train,X_test_scaled,y_test,\
                          Model,opt_a, 0.5, 5,10)

print('training error is %d ;testing error is %d' %(train_e,test_e))
'''