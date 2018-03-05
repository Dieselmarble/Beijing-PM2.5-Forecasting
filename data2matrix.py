#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:16:31 2018

@author: kevin
"""
import numpy as np
import pandas as pd

def data2matrix():
    df2 = pd.read_csv('PRSA_data.csv')#.as_matrix()
    df = df2.copy()
    #use previous pm2.5 as an explicit feature
    df['pm_today'] = df2['pm2.5']
    #shift tomorrow's pm2.5 data into today's row
    #df['pm2.5'] = df['pm2.5'].shift(-24) 
    #remove indexing in features 
    df = df.drop(['No'], axis=1)
    #remove year in features
    df = df.drop(['year'], axis=1)
    df = df.dropna(how='any')
    
    time = np.array(range(9,24))
    for j in range(len(time)):
        #romve data not at 8am
        df = df[df.hour != time[j]] 
    
    #produce an numpy array
    data = np.matrix(df) 
    N = len(data)
    #Wind direction is on column 8 
    p = 7
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
           
    del N, df, df2, i
    
    return data
    
if __name__ == '__main__':
    data2matrix()
    

#csv = np. genfromtxt('PRSA_data.csv',dtype=float, delimiter=",")






    