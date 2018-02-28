#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:31:15 2018

@author: kevin
"""

from sklearn import preprocessing
import numpy as np

X_scaled = preprocessing.scale(data)
#to exame
#X_scaled.mean(axis=0)