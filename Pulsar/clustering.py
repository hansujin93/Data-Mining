# -*- coding: utf-8 -*-

## 1. Hierarchical clustering ##
def dist_one_pair(arr1, arr2):
    sum = 0
    for i in range(len(arr1)):
        sum += (arr1[i]-arr2[i])**2
    sum = math.sqrt(sum)
    return sum


def euc_distance(data):
    data = np.array(data)
    rows = data.shape[0]
    result = np.zeros((rows, rows))

    for i in range(rows):
        for j in range(rows):
            result[i][j] = dist_one_pair(data[i],data[j])
    return result
    
