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

for i in range(1,15): # in range 1 - 7 
    
    #df['day_%d' %i] = df2['day'].shift(23+i)
    #df['hour_%d' %i] = df2['hour'].shift(23+i)
    df['pm2.5_%d' %i] = df2['pm2.5'].shift(23+i)
    df['DEWP_%d' %i] = df2['DEWP'].shift(23+i)
    df['TEMP_%d' %i] = df2['TEMP'].shift(23+i)
    df['PRES_%d' %i] = df2['PRES'].shift(23+i)
    df['cbwd_%d' %i] = df2['cbwd'].shift(23+i)
    df['Iws_%d' %i] = df2['Iws'].shift(23+i)
    df['Is_%d' %i] = df2['Is'].shift(23+i)
    df['Ir_%d' %i] = df2['Ir'].shift(23+i)


#time = np.array(range(9,24))
#for j in range(len(time)):
    #romve data not at 8am
#    df = df[df.hour != time[j]] 
    
#df = df[df.hour ==8] 
#shift tomorrow's pm2.5 data into today's row
df['pm2.5_to_predict'] = df['pm2.5'].shift(-24) 
df = df.drop(['pm2.5'], axis=1)
#produce an numpy array
df = df.dropna(how='any')
data = np.matrix(df) 
N = len(data)
#Wind direction is on column 8 

for i in range (N):
        for p in range(6,123,8):
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





