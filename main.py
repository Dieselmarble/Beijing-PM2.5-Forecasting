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
import scipy.interpolate as interp

def linear_method():
    errors = []
    coefs = []
    train_e=[]
    alphas = np.logspace(-8, 8, 100)
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
        if errors[i] < mini:
            mini = errors[i]
            index = i        
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
    plt.title('Ridge training and CV error versus the regularization')
    plt.show()
    
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1]) #reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients versus the regularization')
    plt.axis('tight')
    plt.show()
    train_e, test_e = testing(X_train_scaled,y_train,X_test_scaled,y_test,\
                              Model,opt_a,0.5, 0, 0)
    print('training error is %d ;testing error is %d' %(train_e,test_e))

def nlinear_method():    
    C_range = np.logspace(-2, 2, 5)
    ep_range = np.logspace(-2, 2, 5)
    errors=[]
    parc=[]
    parep=[]
    i = 0
    j = 0
    e_mesh = np.zeros(shape=(len(C_range),len(ep_range)))
    t_mesh = np.zeros(shape=(len(C_range),len(ep_range)))
    for C in C_range:
        for ep in ep_range:
            #cross_valid(feature, y, name, a, l1, C, epsilon)
            mse, te, coef  = cross_valid \
            (X_train_scaled, y_train, Model, 1, 0.5, C, ep)
            e_mesh[i][j] = mse
            t_mesh[i][j] = te
            errors.append(mse)
            parc.append(C)
            parep.append(ep)
            print('almost done %d' %j)
            j+=1
        j=0    
        i+=1
    mini = errors[0]
    for k in range(len(errors)):
        if errors[k] < mini:
            mini = errors[k]
            index = k
    opt_c = parc[index]
    opt_ep = parep[index]
    print('C value %d, epsilon value %d has lowest error of %d' \
          %(opt_c,opt_ep, mini))
    
    # Plot the surface
    plt.imshow(e_mesh, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('epsilon')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(ep_range)), ep_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('CV error SVM')
    plt.show()
    
    plt.imshow(t_mesh, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('epsilon')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(ep_range)), ep_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Train error SVM')
    plt.show()
    train_e, test_e = testing(X_train_scaled,y_train,X_test_scaled,y_test,\
                              Model,0,0.5, opt_c,opt_ep)
    print('training error is %d ;testing error is %d' %(train_e,test_e))

if __name__ == '__main__': 
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
    Model = 'SVM'
    nlinear_method()
