# -*- coding: utf-8 -*-
import numpy as np
import random
import math
from random import randint

## 1. One-hot encoding ##
def multiple(l,n):
    k = []
    while len(k)<n:
        k.append(l)
    k = np.array(k)
    return k

def onehot_encode(data, which_column): 
    l = []
    newdata = data.copy()
    for i in range(len(data)):
        l.append(data[i][which_column])
    uniq = np.unique(l)
    new = multiple([0]*len(uniq),len(data))
    for p in range(len(l)):
        for idx in range(len(uniq)):
            if l[p] == uniq[idx]:
                new[p][idx] = 1
    newdata = np.delete(newdata, which_column, 1) 
    newdata = np.concatenate((newdata, new), axis = 1) 
    return newdata



## 2. normalisation ##
def normalise(xtrain, xtest):
    norm_xtrain = xtrain.copy()
    norm_xtest = xtest.copy()

    min_list = []
    max_list = []

    rows = xtrain.shape[0]
    cols = xtrain.shape[1]

    for j in range(cols):
        max_value = np.amax(xtrain[:, j])
        max_list.append(max_value)
        min_value = np.amin(xtrain[:, j])
        min_list.append(min_value)

        for i in range(rows):
            norm_xtrain[i, j] = (xtrain[i, j] - min_value) / (max_value - min_value)

    for j in range(xtest.shape[1]):
        for i in range(xtest.shape[0]):
            norm_xtest[i][j] = (xtest[i][j] - min_list[j]) / (max_list[j] - min_list[j])

    return norm_xtrain, norm_xtest



## 3. Sampling ##
''' 1. undersampling '''
def undersample(x, y, k, l):
    data = np.concatenate((x, y.reshape(len(y),1)), axis=1)
    newData = np.array([])
    labelLPositions = []
    for rowPos in range(len(data)):
        if (y[rowPos] == l):
            labelLPositions.append(rowPos)                 
    toEliminate = random.sample(labelLPositions, k)       
    for rowPos in range(len(data)):
        if (rowPos not in toEliminate):
            if (len(newData) == 0):
                newData = data[rowPos, :].copy()
            else:
                newData = np.vstack((newData, data[rowPos, :]))  
    x = newData[:,:10]    # since 10th is response variabel
    y = newData[:,10]
    y = y.reshape(len(y),1)
    return x, y

''' 2. oversampling '''
def oversample(x, y, k, l):
    data = np.concatenate((x, y.reshape(len(y),1)), axis=1)
    newData = data.copy()
    labelLPositions = []
    for rowPos in range(len(data)):
        if (y[rowPos] == l):
            labelLPositions.append(rowPos)

    for n in range(k):
        selectedItemIndex = labelLPositions [random.randint(0,len(labelLPositions)-1)]
        selectedItem = data[selectedItemIndex]
        newData = np.vstack((newData, selectedItem))

    x = newData[:,:10]    # since 10th is response variabel
    y = newData[:,10]
    y = y.reshape(len(y),1)
    return x, y

''' 3. smote '''
def dist_one_pair(arr1, arr2):
    sum = 0
    for i in range(len(arr1)):
        sum += (arr1[i]-arr2[i])**2
    sum = math.sqrt(sum)
    return sum

def dist_group(group, point, k):
    dist_list = []
    for i in group:
        dist_list.append(dist_one_pair(i, point))
    nn = np.argsort(dist_list)                # ordered by closest data index ~ farthest data index
    return nn[1:k+1]                          # return index (but the closest [0] would be the point itself so exclude

def get_between_point(p1,p2,alpha):
    random_point = [0] * len(p1)
    for l in range(len(p1)):
        random_point[l] = alpha*p1[l] + (1-alpha)*p2[l]
    return random_point

def smote(x, y, n, l, k, except_column_list):   # k: number of neighbours that we require
    which_label = []
    label_data = y.copy()
    xdata = x.copy()
    for i in range(len(x)):
        if y[i] == l:
            which_label.append(x[i])                      # append data points (not index)
    fake_samples = []
    for iteration in range(n):                            # iterate for n times: set by users
        neighbors = []
        p = which_label[randint(0,len(which_label)-1)]    # pick random point in which_label to be a standard of making fake samples
        # print("pick p as", p)
        for idx in dist_group(which_label, p, k):
            neighbors.append(which_label[idx])            # append neighbors of random points which are  within k from certain point
        random_j = neighbors[randint(0,len(neighbors)-1)] # pick random neighbor
        # print("random_j as ", random_j)
        alpha = np.random.random()
        fake_sample = get_between_point(random_j, p, alpha)
        for ec in except_column_list:
            if alpha >= 0.5:
                fake_sample[ec] = random_j[ec]   
            else:
                fake_sample[ec] = p[ec]          
        # print('after revise 7,8,9 ', fake_sample)
        fake_samples.append(fake_sample)
    label_data = label_data.reshape(len(label_data),1)
    sample_label = np.array([l]*len(fake_samples))
    sample_label = sample_label.reshape(len(sample_label),1)
    concat_label = np.concatenate((label_data,sample_label), axis=0)
    concat_x = np.concatenate((xdata,fake_samples), axis=0)
    return concat_x, concat_label
