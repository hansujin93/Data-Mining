import numpy as np

def count_missing_value(data):
    feature_number = data.shape[1]
    per_feature = [0]*feature_number
    for j in range(feature_number):
        for i in range(len(data)):
            if data[i][j]=='nan':
                per_feature[j] += 1
    return per_feature
    
## 1. substitute to a new feature named 'unknown' ##
def na_to_unknown(data, which_column):
    newdata = data.copy()
    for i in range(len(data)):
        if 'nan' in data[i,which_column]:
            newdata[i, which_column] = "unknown"
    return newdata
    
## 2. delete column ##
def delete_na(data, which_column):
    newdata = data.copy()
    newdata = np.delete(data, which_column,1) # 1: delete column
    return newdata
    
## 3. delete row ##
def remove_narows(data, labeldata):
    newdata = data.copy()
    new_labeldata = labeldata.copy()
    where_na = []
    for i in range(len(data)):
        if ('nan' in data[i]):
            where_na.append(i)

    newdata = np.array([row for num, row in enumerate(newdata) if num not in where_na])
    new_labeldata = np.array([row for num, row in enumerate(labeldata) if num not in where_na])
    return newdata, new_labeldata
   
## 4. mode ##
def extract_mode(data, col):
    value, count = np.unique(data[:, col], return_counts=True)
    mode_value = value[np.argmax(count)]
    return mode_value

def replace(data, col, before, after):
    newdata = data.copy()
    for i, element in enumerate(newdata):
        if element[col] == before:
            newdata[i][col] = after
    return newdata
