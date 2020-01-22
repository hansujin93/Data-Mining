# -*- coding: utf-8 -*-

## 1. LDA ##
def meanfeatures(data):
    mean_list = []
    for i in range(data.shape[1]):
        mean_list.append(np.mean(data[:,i]))
    return mean_list

def divide_by_label(data, label_data, label):
    labeled_X = []
    for i in range(len(data)):
        if label_data[i][0] == label:
            labeled_X.append(data[i])
    labeled_X = np.array(labeled_X)
    return labeled_X
    
def lda_func(x, y, k):
    # calculation of within class scatter matrix
    X_train_0 = divide_by_label(x, y, 0)
    X_train_1 = divide_by_label(x, y, 1)

    meanfeature_0 = np.array(meanfeatures(X_train_0))
    meanfeature_1 = np.array(meanfeatures(X_train_1))
    overallmean = meanfeatures(x)

    sw = np.zeros((x.shape[1], x.shape[1]))
    for p in range(len(X_train_0)):
        diff = (X_train_0[p, :] - meanfeature_0).reshape(x.shape[1], 1)

        sw += diff.dot(diff.T)
    for q in range(len(X_train_1)):
        diff = (X_train_1[q, :] - meanfeature_1).reshape(x.shape[1], 1)
        sw += diff.dot(diff.T)

    meanfeature_0 = meanfeature_0.reshape(1, (len(meanfeature_0)))
    meanfeature_1 = meanfeature_1.reshape(1, (len(meanfeature_1)))
    means = np.concatenate((meanfeature_0, meanfeature_1), axis=0)

    sb = np.zeros((X_train_0.shape[1], X_train_0.shape[1]))
    for c in range(2):  # since it is binary classification
        diff = (means[c, :] - overallmean).reshape(x.shape[1], 1)
        sb += len(X_train_0) * diff.dot(diff.T)

    inv_sw = linalg.inv(sw)
    eigvals, eigvectors = linalg.eig(inv_sw.dot(sb))
    ordered_eigvectors = np.empty(eigvectors.shape)
    tmp = eigvals.copy()

    for i in range(len(eigvectors)):
        maxvalue = float("-inf")
        maxvalue_pos = -1
        for n in range(len(eigvectors)):
            if (tmp[n] > maxvalue):
                maxvalue = tmp[n]
                maxvalue_pos = n
        ordered_eigvectors[:, i] = eigvectors[:, maxvalue_pos]
        tmp[maxvalue_pos] = float("-inf")

    project_matrix = ordered_eigvectors[:, 0:k]

    return project_matrix

def lda_transform(data, project_matirx):
    ldadata_transformed = data.dot(project_matirx)
    return ldadata_transformed

def get_ldalist_func(x,y):
    list0 = []
    list1 = []
    for i in range(len(x)):
        if y[i] == 0:
            list0.append(x[i])
        else:
            list1.append(x[i])
    list0 = np.array(list0)
    list1 = np.array(list1)
    return list0, list1

def lda_plot_2d(ldx_xtrain, lda_xtest, ytrain, ytest):
    list0, list1 = get_ldalist_func(ldx_xtrain, ytrain)
    test0, test1 = get_ldalist_func(lda_xtest, ytest)

    plt.figure(figsize=(6, 4))
    plt.plot(list0[:, 0], list0[:, 1], "r.")
    plt.plot(list1[:, 0], list1[:, 1], "g.")
    plt.plot(test0[:, 0], test0[:, 1], "rx")
    plt.plot(test1[:, 0], test1[:, 1], "gx")
    plt.xlabel("1st Principal Component ")
    plt.ylabel("2nd Principal Component ")
    plt.show()

def lda_plot_3d(ldx_xtrain, lda_xtest, ytrain, ytest):
    list0, list1 = get_ldalist_func(ldx_xtrain, ytrain)
    test0, test1 = get_ldalist_func(lda_xtest, ytest)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    train_class0_x = [i for i in list0[:, 0]]  # train, class=0
    train_class0_y = [i for i in list0[:, 1]]
    train_class0_z = [i for i in list0[:, 2]]
    train_class1_x = [i for i in list1[:, 0]]  # train, class=1
    train_class1_y = [i for i in list1[:, 1]]
    train_class1_z = [i for i in list1[:, 2]]
    test_class0_x = [i for i in test0[:, 0]]  # test, class=0
    test_class0_y = [i for i in test0[:, 1]]
    test_class0_z = [i for i in test0[:, 2]]
    test_class1_x = [i for i in test1[:, 0]]  # test, class=1
    test_class1_y = [i for i in test1[:, 1]]
    test_class1_z = [i for i in test1[:, 2]]

    ax.scatter(train_class0_x, train_class0_y, train_class0_z, c='r', marker='o')
    ax.scatter(train_class1_x, train_class1_y, train_class1_z, c='g', marker='o')
    ax.scatter(test_class0_x, test_class0_y, test_class0_z, c='r', marker='x')
    ax.scatter(test_class1_x, test_class1_y, test_class1_z, c='g', marker='x')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.show()

def lda_predict(xtrain, ytrain, xtest, k):
    proj_matrix = lda_func(xtrain, ytrain, k)

    lda_xtrain = lda_transform(xtrain, proj_matrix)
    lda_xtest = lda_transform(xtest, proj_matrix)

    trans0 = divide_by_label(lda_xtrain,ytrain,0)
    trans1 = divide_by_label(lda_xtrain,ytrain,1)
    mean0 = np.array(meanfeatures(trans0))
    mean1 = np.array(meanfeatures(trans1))

    pred = []

    for i in range(len(lda_xtest)):
        # when transformed test x is closer to the mean of transformed train x(class0), it will be classified as 0
        if dist_one_pair(mean0,lda_xtest[i,:]) < dist_one_pair(mean1,lda_xtest[i,:]):
            pred.append(0)
        else:
            pred.append(1)

    return pred
