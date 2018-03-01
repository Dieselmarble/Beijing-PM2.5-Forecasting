#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:16:31 2018

@author: kevin
"""
import numpy as np
import pandas as pd

def data2matrix():
    read = pd.read_csv('PRSA_data.csv')#.as_matrix()
    #shift tomorrow's pm2.5 data into today's ros
    df = read.dropna(how='any')
    #romve data not at 8am
    df = df[df.hour == 8]
    #remove indexing in features 
    df = df.drop(['No'], axis=1)
    p = 8
    #shift tomorrow's pm2.5 data into today's row
    read['pm2.5'] = read['pm2.5'].shift(-1)
    # to change use .astype() 
    #produce an numpy array
    
    data = np.matrix(df) 
    N = len(data)
    for i in range (N):
        if data[i,p] == 'NE':
            data[i,p] = 1
        elif data[i,p] == 'SE':
            data[i,p] = 2
        elif data[i,p] == 'NW':
            data[i,p] = 4
        elif data[i,p] == 'cv':
            data[i,p] = 0
    data = data.astype(float)        
            
    del N, df,i,read
    
    return data
    
if __name__ == '__main__':
    data2matrix()
    

#csv = np. genfromtxt('PRSA_data.csv',dtype=float, delimiter=",")






    