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
    #shift tomorrow's pm2.5 data into today's row
        
    read['pm2.5'] = read['pm2.5'].shift(-24)
    
    df = read.dropna(how='any')
   
    #remove indexing in features 
    df = df.drop(['No'], axis=1)
    #df = df.drop(['year'], axis=1)
    
    #Wind direction is on column 8 
    p = 8
   
    #df2 = df.copy()
    #romve data not at 8am
    #df2 = df2[df2.hour == 8]
    #data2 = np,matrix(df2)
    
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
    
    # to change use .astype()      
    data = data.astype(float)
    #data2 = data2.astype(float)         
           
    del N, df, i, read
    
    return data
    
if __name__ == '__main__':
    data2matrix()
    

#csv = np. genfromtxt('PRSA_data.csv',dtype=float, delimiter=",")






    