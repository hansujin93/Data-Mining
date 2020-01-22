# -*- coding: utf-8 -*-
from scipy.spatial.distance import cdist
import numpy as np

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



## 2. Decision Tree ##
''' 1) Entropy based '''
def entropy(col):
    elements, counts = np.unique(col,return_counts=True)  # extract unique element to 'elements'
                                                          # / extract number of frequencies for each element to 'counts'
    entropy = -1 * np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))
                      for i in range(len(elements))])     # entropy equation
    return entropy

# function of calculating information gained of dataset/subset
def infogain(data, split_feature, target="class"):
    # split_feature: splitting point of feature where I want to test the information gained based on
    # target: class(0/1)
    # calculate baseline entropy
    total_entropy = entropy(data[split_feature])
    vals, counts = np.unique(data[split_feature], return_counts=True)

    # calculate weighted entropy based on vals and target(class)
    # if the weighted entropy is small, it gives more information so to be chosen as 'interior node'
    weighted_entropy = np.sum( [(counts[i] / np.sum(counts)) *
         entropy(data.where(data[split_feature] == vals[i]).dropna()[target] )  # get entropy of the target(class) for each values in split_feature
         for i in range(len(vals))  ]) # store as a list to compare the value afterwards
    info_gain = total_entropy - weighted_entropy
    return info_gain

# contains every case of stopping point of decision tree
# 1) when all rows have same class (> no need to split more)
# 2) when no more features to split further left (> one independent feature left and class still have either 0 or 1)
# 3) when no more rows left
def ID3(data, original_data, features, target="class",parent_node_class=None):
    # data: dynamic data (keep reducing columns after one split
    # original_data: static data / real data set
    ####### this is for stopping point of splitting #######
    if len(np.unique(data[target])) <= 1:   # when the stopping point = 1),
        return np.unique(data[target])[0]   # return that class
    elif len(data) == 0:                    # when the stopping point = 3),
        return np.unique(original_data[target])[np.argmax(np.unique(original_data[target],return_counts=True)[1])]
        # return mode class of original data set
    elif len(features) == 0:   # this cannot be happened in the first place, but later can be...
        return parent_node_class

    ###### this is for building tree #######
    else:
        # class of parent_node = frequent class in parent node
        parent_node_class = np.unique(data[target])[np.argmax(np.unique(data[target],return_counts=True)[1])]
        info_list = [infogain(data, feature, target) for feature in features] # calculate gained information for every element in features(>to be split)
        best_feature = features[np.argmax(info_list)]  # this becomes an interior node
    # store feature that becomes a standard to split (> that becomes an interior node)
        tree = {best_feature:{}}
    # since best_feature is already used, remove it from features to distinguish next possible split nodes
        features = [i for i in features if i!=best_feature]
    # it checks every value in the feature
        for value in np.unique(data[best_feature]):
            subdata = data.where(data[best_feature]==value).dropna()
            # recursively build tree while keeping reducing features to split
            subtree = ID3(subdata, original_data, features, target, parent_node_class)
            tree[best_feature][value] = subtree # store tree inside tree inside tree...
    return tree

def DC_predict(query, tree, default = 1):
    for key in list(query.keys()):  # for every features
        if key in list(tree.keys()):  # this is for exception when the value in test set is new that didn't exist in train set
            try:
                result = tree[key][query[key]] 
            except:
                return default  # in case of value which exist in test set but not in train set
                                # tree[key] = value of feature that is a splitting point
            if isinstance(result,dict):  # if result is of dictionary data type, it means that there still exists tree below, so recursively check
                return DC_predict(query, result)
            else:
                return result  # if it is not, then the result is just a value (predicted)

def DC_test(data, tree):
    queries = data.iloc[:,:-1].to_dict(orient="records")
    pred_value = []
    for i in range(len(data)):
        pred_value.append(DC_predict(queries[i],tree))
    return pred_value

''' 2) Gini index based '''
class DecisionTree:

    def __init__(self, max_depth=None, min_node=None):
        self.max_depth = max_depth
        self.min_node = min_node

    def test_split(self, index, value, data):
        left, right = list(), list()
        for row in data:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # calculate gini index for a split point
    # classes = unique number of class
    def gini_index(self, groups, classes):
        total_n = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups: 
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for value in classes: 
                p = [element[-1] for element in group].count(value) / size
                score += p * p
            gini += (1.0 - score) * (size / total_n)
        return gini

    def get_best_split(self, data):
        unique_class = list(set(row[-1] for row in data))
        best_idx, best_value, best_score, best_groups = 999, 999, 999, None   # hard-coding
        for idx in range(len(data[0]) - 1):  # exclude -1th which indicates class
            for row in data:
                # start with left = 0 since row[idx] = row[idx]
                groups = self.test_split(idx, row[idx], data)
                gini = self.gini_index(groups, unique_class)

                if gini < best_score:
                    best_idx, best_value, best_score, best_groups = idx, row[idx], gini, groups
        return {'index': best_idx, 'value': best_value, 'groups': best_groups}

    # terminal nodes
    # 1) when maximum tree depth is reached
    # w) when less than minimum node will be occurred by splitting
    # to deal with this sort of terminal points, set separate function for these cases
    def terminal(self, group):
        classes = [row[-1] for row in group]
        # return most frequently appeared class
        return max(set(classes), key=classes.count)

    def split(self, node, depth):
        left, right = node['groups']
        del (node['groups'])  # no more need to be stored. Instead, it should be renewed for further splitting (for additional children leaf)

        # if either group is empty, cannot split but just return most frequent class among them
        if not left or not right:
            node['left'] = node['right'] = self.terminal(left + right)
            return

        if depth >= self.max_depth:
            node['left'], node['right'] = self.terminal(left), self.terminal(right)
            return

        if len(left) <= self.min_node:
            node['left'] = self.terminal(left)
        else:
            # if there are enough samples, continue recursive splitting
            node['left'] = self.get_best_split(left)
            # recursive function (since leaf gets smaller and smaller as splitting continues,
            # this will go to terminal node at the end
            self.split(node['left'], depth + 1)

        if len(right) <= self.min_node:
            node['right'] = self.terminal(right)
        else:
            # if there are enough samples, continue recursive splitting
            node['right'] = self.get_best_split(right)
            # recursive function (since leaf gets smaller and smaller as splitting continues,
            # this will go to terminal node at the end
            self.split(node['right'], depth + 1)

    def build_tree(self, data):
        # first splitting point
        root = self.get_best_split(data)
        self.split(root, 1)
        return root

    def predict(self, node, testdata):
        if testdata[node['index']] < node['value']:  # go to left node as defined in test_split
            if isinstance(node['left'], dict):
                # predict recursively
                return self.predict(node['left'], testdata)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                # predict recursively
                return self.predict(node['right'], testdata)
            else:
                return node['right']
