# -*- coding: utf-8 -*-
import numpy as np
import scipy.cluster as sc
import scipy.spatial.distance as sd

## 1) Hierarchical clustering ##
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

dist_data_x = euc_distance(x)
condensed_dist = sd.squareform(dist_data_x)



## 2) K means ##
def which_cluster(point, centroids):
    position = -1
    ans = []
    for i in range(len(centroids)):
        d = dist_one_pair(point, centroids[i])
        ans.append(d)
    for k in range(len(ans)):
        if ans[k] == min(ans):
            position = k
    return position

# For error calculation between old, new centroids (How much the center has moved)
def total_dist(c_1, c_2):
    total = 0
    for i in range(len(c_1)):
        total += dist_one_pair(c_1[i],c_2[i])
    total = math.sqrt(total)
    return total

def kmeans(data, k, C, threshold = 0.01, max_iter = 200):
    error = float("inf")
    clusters = np.zeros(len(data))
    cnt = 0
    while error > threshold:                             # Keep running while loop before error gets lower than threshold
        if cnt > max_iter:                               # If it cannot reduce error up to max_iter, break the loop and return latest result
            break
        else:
            for i in range(len(data)):
                clusters[i] = which_cluster(data[i], C)  # Decide every points which centroid is closer 
            C_old = C.copy()                             # For updating centroid, remember the old centroid
            for j in range(k):
                points = np.array([])
                for ind in range(len(data)):
                    if clusters[ind] == j:
                        if len(points) == 0:
                            points = data[ind].copy()    # Store the points in each cluster
                        else:
                            points = np.vstack((points, data[ind]))
                C[j] = np.mean(points, axis=0)          # generate new centroid based on mean in each cluster
            error = total_dist(C, C_old)
            cnt += 1
    print("Result of centroid: \n", C)
    print("How many iterations: ", cnt)
    return clusters, cnt
