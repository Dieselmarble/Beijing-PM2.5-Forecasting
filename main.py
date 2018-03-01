#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:11:24 2018

@author: kevin
"""

import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from data2matrix import data
from data2matrix import data2matrix
from cross_valid import cross_valid
from normalisation import normalise
from ridge_regression import ridge

def main():
    #import data from csv file to a matirx
    data = data2matrix()
    #split data into training and test set
    train, test = train_test_split(data, test_size=0.2)
    #normalise training set
    #train_scaled = preprocessing.scale(train)
    #nomalise test set with the mean and std from training set
    
    train_scaled, test_scaled = normalise(train, test)
    #to exam
    #new_mean = train_scaled.mean()
    #new_std = train_scaled.std()
    
    #Cross validation splitter
    (train_set,valid_set)=cross_valid(train)
    
    #PM2.5 for training
    PM25_train = train_scaled[:,9]
    
    #Do ridge regression return coefficients and error 
    coef,mse_train = ridge(train_scaled,PM25_train)

    del data


if __name__ == '__main__':
    main()