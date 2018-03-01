#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:25:36 2018

@author: kevin
"""
import numpy as np
import pandas as pd

read = pd.read_csv('PRSA_data.csv')#.as_matrix()
#shift tomorrow's pm2.5 data into today's ros
df = read.dropna(how='any')
#romve data not at 8am
df = df[df.hour == 8]
df = df.drop(['No'], axis = 1)
#shift tomorrow's pm2.5 data into today's row
read['pm2.5'] = read['pm2.5'].shift(-1)
# to change use .astype() 
#produce an numpy array
data = np.matrix(df) 

N = len(data)
i=0;
for i in range (N):
    if data[i,8] == 'NE':
        data[i,8] = 1
    elif data[i,8] == 'SE':
        data[i,8] = 2
    elif data[i,8] == 'NW':
        data[i,8] = 4
    elif data[i,8] == 'cv':
        data[i,8] = 0
        
ddf = data.astype(float)        
        
#del N, df,i,read
    

    

