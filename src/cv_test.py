from sklearn import svm
import sklearn.cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import svm
import os
import time
import numpy as np

from load_data import *
from dualksvm import *
from stocksvm import *


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

    perc = 0.5
    print '--------------------------------------------------------------------'
    print "usps dataset, size=%d, dim=%d, %2d%% for training" % (x.shape[0], x.shape[1], 100 * perc)
    lmda_list = (np.power(2.0, range(-10, 5)) / 1000).tolist()
    # lmda_list = [0.02]
    gamma_list = (np.power(2.0, range(-10, 3))).tolist()
    # gamma_list = [0.03]
    n_folds = 4
    kf = cv.KFold(len(y), n_folds=n_folds, indices=False)
    k = 0
    train_err = np.zeros((len(lmda_list), len(gamma_list), n_folds))
    test_err = np.zeros((len(lmda_list), len(gamma_list), n_folds))

    for k, (train, test) in enumerate(kf):
        ntr = len(train)
        print "kfold: %d" % k
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        mM_scale = preprocessing.MinMaxScaler((-1, 1))
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
    np.save('../output/cv_result_usps.npy', avg_test_err)
    X, Y = np.meshgrid(gamma_list, lmda_list)
    plt.figure()
    plt.contour(X, Y, np.mean(test_err, axis=2))
    plt.xlabel('gamma')
    plt.ylabel('lambda')

    color_list = ['b', 'r', 'g', 'c', 'm']
    marker_list = ['o', 'x', '>', 's', '^']
    plt.figure()
    for i, (c, mk) in enumerate(zip(color_list, marker_list)):
        plt.semilogx(gamma_list, avg_test_err[i, :], c + mk + '-', label='lmda=%f' % lmda_list[i])
    plt.legend(bbox_to_anchor=(0, 1.17, 1, .1), loc=2, ncol=2, mode="expand", borderaxespad=0)
    plt.xlabel('gamma')
    plt.ylabel('test error rate')

    # plt.show()


    # if __name__ == '__main__':
    # profile_usps(8)
    # plot_convergence(8)