#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:25:36 2018

@author: kevin
"""
import numpy as np
import pandas as pd


df2 = pd.read_csv('PRSA_data.csv')#.as_matrix()
#df2 = df2.dropna(how='any')
df = df2.copy()

#remove indexing in features 
df = df.drop(['No'], axis=1)
df = df.drop(['year'], axis=1)
df = df.drop(['month'], axis = 1)
df = df.drop(['day'], axis=1)
df = df.drop(['Is'], axis = 1)
df = df.drop(['Ir'], axis = 1)

for i in range(1,8): # in range 1 - 7 

    df['pm2.5_%d' %i] = df2['pm2.5'].shift(i)
    df['DEWP_%d' %i] = df2['DEWP'].shift(i)
    df['TEMP_%d' %i] = df2['TEMP'].shift(i)
    df['PRES_%d' %i] = df2['PRES'].shift(i)
    df['cbwd_%d' %i] = df2['cbwd'].shift(i)
    df['Iws_%d' %i] = df2['Iws'].shift(i)


df = df[df.hour ==8] 
#shift tomorrow's 8 am pm2.5 data into today's row
df['pm2.5_to_predict'] = df['pm2.5'].shift(-1) 
df = df.drop(['hour'], axis=1)
#produce an numpy array
df = df.dropna(how='any')
data = np.matrix(df) 
N = len(data)
#Wind direction is on column 8 

for i in range (N):
        for p in range(4,49,6):
            if data[i,p] == 'NE':
                data[i,p] = 1
            elif data[i,p] == 'SE':
                data[i,p] = 2
            elif data[i,p] == 'NW':
                data[i,p] = 4
            elif data[i,p] == 'cv':
                data[i,p] = 0
            
# to change use .astype() 
data = data.astype(float)
#data2 = data2.astype(float)         
       
del N, i, df2, p





