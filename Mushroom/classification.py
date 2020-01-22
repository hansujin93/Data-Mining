# -*- coding: utf-8 -*-


## 1. KNN ##
def knn_predict(train_x, train_y, test, k):
    dist = cdist(test, train_x)
    pred = []  
    for i in range(len(test)):
        neighbors = np.array(dist[i, :])
        sorted_idx = np.argsort(neighbors)
        sorted_idx_k = sorted_idx[:k]  # only kth closest items's index
        freq_label = []
        for klab in sorted_idx_k:
            freq_label.append(train_y[klab])  # append labels of k closest neighbors in order
        values, count = np.unique(freq_label, return_counts=True)
        pred.append(values[np.argmax(count)])  # predict test[i] for the frequent label in freq_label
    return pred
