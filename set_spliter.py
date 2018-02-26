#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:25:36 2018

@author: kevin
"""

import numpy as np
from sklearn.model_selection import train_test_split
#from data2matrix import data
from data2matrix import data2matrix

data = data2matrix()
train, test = train_test_split(data, test_size=0.2)



