# -*- coding: utf-8 -*-

## Feature Selection ##
''' 1) correlation based '''
import numpy as np
import math
import pandas as pd

def mean_func(data_array):
    total = 0
    for i in data_array:
        total += float(i)
    mean = total / len(data_array)
    return mean

def sd_func(data_array):
    mean = mean_func(data_array)
    dev = 0.0
    for i in range(len(data_array)):
        dev += (data_array[i]-mean)**2
    dev = math.sqrt(dev)
    return dev

def corr_func(data_array_1, data_array_2):
    mean_1 = mean_func(data_array_1)
    mean_2 = mean_func(data_array_2)
    sd_1 = sd_func(data_array_1)
    sd_2 = sd_func(data_array_2)
    r = 0.0
    for k in range(len(data_array_1)):
        r += (data_array_1[k]-mean_1)*(data_array_2[k]-mean_2)
    corr_cf = round(r / (sd_1*sd_2),4)
    return corr_cf
    
def corr_matrix(data):
    corr_m = np.zeros([data.shape[1], data.shape[1]])
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            corr_by_features = corr_func(data[:,i],data[:,j])
            corr_m[i][j] = corr_by_features
    corr_m = pd.DataFrame(corr_m)
    return corr_m


''' 2) permutation based '''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
import eli5

X_train, X_valid, y_train, y_valid = train_test_split(data_x, data_y, test_size=0.3, random_state=123)
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
perm = PermutationImportance(model, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm)
