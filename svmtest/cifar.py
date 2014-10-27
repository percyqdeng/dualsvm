import numpy as np
import os
import pickle
import sklearn.preprocessing as preprocessing
import sklearn.cross_validation as cv
import scipy.linalg as linalg
from sklearn import svm
from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
print "-----------------------load cifar data --------------------------"
data_name = 'cifar'
xtrain = None
ytrain = None
for i in range(1,6):
    data_path = os.path.join('..', 'dataset','cifar', 'cifar-10-batches-py', 'data_batch_%s' % i)
    data = pickle.load(open(data_path, 'r'))
    if xtrain is not None:
        xtrain = np.vstack((xtrain, np.asarray(data['data'], dtype=float)))
        ytrain = np.hstack((ytrain, np.asarray(data['labels'], dtype=float)))
    else:
        xtrain = np.asarray(data['data'], dtype=float)
        ytrain = np.asarray(data['labels'], dtype=float)


data_path = os.path.join('..', 'dataset','cifar', 'cifar-10-batches-py', 'data_batch_%s' % i)
data = pickle.load(open(data_path, 'r'))
xtest = np.asarray(data['data'], dtype=float)
ytest = np.asarray(data['labels'], dtype=float)

scalar = preprocessing.StandardScaler().fit(xtrain)
xtrain = scalar.transform(xtrain)
xtest = scalar.transform(xtest)

ytrain[ytrain != 4] = -1
ytrain[ytrain==4 ] = 1
ytest[ytest != 4] = -1
ytest[ytest == 4] = 1

x = np.vstack((xtrain, xtest))
y = np.hstack((ytrain, ytest))
# kernel svm

x_train, x_test, y_train, y_test = cv.train_test_split(x, y, train_size=0.05, random_state=11)
# x = min_max_scaler.fit_transform(x)
xsq = (x_train ** 2).sum(axis=1)
dist = xsq[:, np.newaxis] - 2 * x_train.dot(x_train.T) + xsq[np.newaxis, :]
gamma = 3.0 / np.median(dist)
print "-----------------------train svm ------------------------------------"
clf = svm.SVC(verbose=True, kernel='rbf', gamma=gamma)
clf.fit(x_train, y_train)
pred = clf.predict(x_train)
err_train = zero_one_loss(pred, y_train)
pred = clf.predict(x_test)
err_test = zero_one_loss(pred, y_test)
print 'train error %f ' % (err_train)