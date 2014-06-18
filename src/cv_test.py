from sklearn import svm
import sklearn.cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import os
import time
import numpy as np

from load_data import *
from dualksvm import *
from stocksvm import *
"""
libsvm grid searh
"""

# def plot_convergence(pos_class=3, neg_class=None, random_state=None):
if __name__ == "__main__":
    pos_class = 3
    neg_class = None
    random_state = None
    """
    plot of convergence for usps data on binary classification
    :param pos_class: the digit as positive class
    :param neg_class: the digit as negative class
    :return:
    """
    data = load_usps()
    if neg_class is None:
        x, y = convert_one_vs_all(data, pos_class)
        one_vs_rest = True
    else:
        x, y = convert_binary(data, pos_class, neg_class)
        one_vs_rest = False

    perc = 0.7
    print '--------------------------------------------------------------------'
    print "usps dataset, size=%d, dim=%d, %2d%% for training" % (x.shape[0], x.shape[1], 100 * perc)
    lmda_list = (np.power(2.0, range(-10, 1)) / 1000).tolist()
    # lmda_list = [0.02]
    gamma_list = (np.power(2.0, range(-10, 0))).tolist()
    # gamma_list = [0.03]
    n_folds = 1
    n_iter = 2
    ss = cv.StratifiedShuffleSplit(y, n_iter=n_iter, test_size=1-perc, train_size=None, random_state=0)
    # kf = cv.KFold(len(y), n_folds=n_folds, indices=False)
    # k = 0
    train_err = np.zeros((len(lmda_list), len(gamma_list), n_iter))
    test_err = np.zeros((len(lmda_list), len(gamma_list), n_iter))

    for k, (train, test) in enumerate(ss):
        ntr = len(train)
        print "kfold: %d" % k
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        mM_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_train = mM_scale.fit_transform(x_train)
        x_test = mM_scale.transform(x_test)
        for i, lmda in enumerate(lmda_list):
            for j, gm in enumerate(gamma_list):
                print "check lamda %f, gamma %f" % (lmda, gm)
                clf = svm.SVC(C=lmda * ntr, kernel='rbf', gamma=gm, cache_size=600)
                clf.fit(x_train, y_train)
                pred = clf.predict(x_train)
                train_err[i, j, k] = zero_one_loss(y_train, pred)
                pred = clf.predict(x_test)
                test_err[i, j, k] = zero_one_loss(y_test, pred)
                # k += 1
    avg_test_err = np.mean(test_err, axis=2)
    avg_train_err = np.mean(train_err, axis=2)
    np.save('../output/cv_result_usps.npy', avg_test_err)
X, Y = np.meshgrid(gamma_list, lmda_list)
plt.figure()
plt.contour(np.log(X), np.log(Y), 1-avg_test_err)
plt.xlabel('log gamma')
plt.ylabel('log lambda')
plt.savefig('../output/usps_contour.pdf')

color_list = ['b', 'r', 'g', 'c', 'm']
marker_list = ['o', 'x', '>', 's', '^']
plt.figure()
for i, (c, mk) in enumerate(zip(color_list, marker_list)):
    plt.semilogx(gamma_list, avg_test_err[i, :], c + mk + '-', label='lmda=%f' % lmda_list[i])
plt.legend(bbox_to_anchor=(0, 1.17, 1, .1), loc=2, ncol=2, mode="expand", borderaxespad=0)
plt.xlabel('gamma')
plt.ylabel('usps test error rate')
plt.savefig('../output/usps_test_libsvm.pdf')

plt.figure()
for i, (c, mk) in enumerate(zip(color_list, marker_list)):
    plt.semilogx(gamma_list, avg_train_err[i, :], c + mk + '-', label='lmda=%f' % lmda_list[i])
plt.legend(bbox_to_anchor=(0, 1.17, 1, .1), loc=2, ncol=2, mode="expand", borderaxespad=0)
plt.xlabel('gamma')
plt.ylabel('usps train error rate')
plt.savefig('../output/usps_train_libsvm.pdf')

    # plt.show()


    # if __name__ == '__main__':
    # profile_usps(8)
    # plot_convergence(8)