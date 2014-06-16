
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import neighbors
import os
import time
from load_data import *

n_neighbors = [1,2,4,10, 15,20, 30, 50, 60, 70, 85, 90]
pos_class = 3
neg_class = None
random_state = None

data = load_usps()
if neg_class is None:
    x, y = convert_one_vs_all(data, pos_class)
    one_vs_rest = True
else:
    x, y = convert_binary(data, pos_class, neg_class)
    one_vs_rest = False

n_folds = 4
kf = cv.KFold(len(y), n_folds=n_folds, indices=False)
train_err = np.zeros((len(n_neighbors), n_folds))
test_err = np.zeros((len(n_neighbors), n_folds))
for k, (train, test) in enumerate(kf):
    print 'kfold %d' % k
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
    for i, n in enumerate(n_neighbors):
        print "# of neighbors: %d" % n
        clf = neighbors.KNeighborsClassifier(n_neighbors=n)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        test_err[i, k] = zero_one_loss(pred, y_test)
        pred = clf.predict(x_train)
        train_err[i, k] = zero_one_loss(pred, y_train)

avg_train_err = np.mean(train_err, axis=1)
avg_test_err = np.mean(test_err, axis=1)

plt.figure()
plt.plot(n_neighbors, avg_train_err, 'rx-', label='trainerr')
plt.plot(n_neighbors, avg_test_err, 'bx-', label='tresterr')
plt.show()

