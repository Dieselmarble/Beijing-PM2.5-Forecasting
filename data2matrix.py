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
    #shift tomorrow's pm2.5 data into today's row
    read['pm2.5'] = read['pm2.5'].shift(-1)
    #produce an numpy array
    data = np.matrix(df) 
    N = len(data)
    for i in range (N ):
        if data[i,9] == 'NE':
            data[i,9] = 1
        elif data[i,9] == 'SE':
            data[i,9] = 2
        elif data[i,9] == 'NW':
            data[i,9] = 4
        elif data[i,9] == 'cv':
            data[i,9] = 0
            
    del N, df,i,read
    return data
    
if __name__ == '__main__':
    data2matrix()
    

#csv = np. genfromtxt('PRSA_data.csv',dtype=float, delimiter=",")






    