# -*- coding: utf-8 -*-

## 1. One-hot encoding ##
def multiple(l,n):
    k = []
    while len(k)<n:
        k.append(l)
    k = np.array(k)
    return k

def onehot_encode(data, which_column):   # which_column = 0(sex) in this case
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
    newdata = np.delete(newdata, which_column, 1) # delete the categorical column
    newdata = np.concatenate((newdata, new), axis = 1) # merge column
    return newdata
