
import sklearn.cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from auxiliary.load_data import *
from dsvm.dualksvm import DualKSVM
from dsvm.stocksvm import Pegasos
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


# if __name__ == "__main__":
def run_gamma(x, y):
    perc = 0.6
    n = x.shape[0]
    gamma_list = (np.power(2.0, range(-4, 12))/(n*perc)).tolist()
    n_iter = 2
    train_err_libsvm = np.zeros((len(gamma_list), n_iter))
    test_err_libsvm = np.zeros((len(gamma_list), n_iter))
    train_err_dsvm = np.zeros((len(gamma_list), n_iter))
    test_err_dsvm = np.zeros((len(gamma_list), n_iter))
    train_err_pegasos = np.zeros((len(gamma_list), n_iter))
    test_err_pegasos = np.zeros((len(gamma_list), n_iter))
    ss = cv.StratifiedShuffleSplit(y, n_iter=n_iter, test_size=1-perc, train_size=None, random_state=0)
    for k, (train, test) in enumerate(ss):
        ntr = len(train)
        lmda = 1.0 / ntr
        print "#iter: %d" % k
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        mM_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        x_train = mM_scale.fit_transform(x_train)
        x_test = mM_scale.transform(x_test)
        for j, gm in enumerate(gamma_list):
            print "check lamda %f, gamma %f" % (lmda, gm)
            clf = svm.SVC(C=lmda * ntr, kernel='rbf', gamma=gm, cache_size=600)
            clf.fit(x_train, y_train)

            pred = clf.predict(x_train)
            train_err_libsvm[j, k] = zero_one_loss(y_train, pred)
            pred = clf.predict(x_test)
            test_err_libsvm[j, k] = zero_one_loss(y_test, pred)
            dsvm = DualKSVM(lmda=lmda, gm=gm, kernelstr='rbf', nsweep=ntr/2, b=5, c=1)
            dsvm.fit(x_train, y_train, x_test, y_test, )
            train_err_dsvm[j, k] = dsvm.err_tr[-1]
            test_err_dsvm[j, k] = dsvm.err_te[-1]
            kpega = Pegasos(ntr, lmda, gm, nsweep=2, batchsize=2)
            kpega.train_test(x_train, y_train, x_test, y_test)
            train_err_pegasos[j, k] = kpega.err_tr[-1]
            test_err_pegasos[j, k] = kpega.err_te[-1]
    avg_train_err_libsvm = np.mean(train_err_libsvm, axis=1)
    avg_test_err_libsvm = np.mean(test_err_libsvm, axis=1)
    avg_train_err_dsvm = np.mean(train_err_dsvm, axis=1)
    avg_test_err_dsvm = np.mean(test_err_dsvm, axis=1)
    avg_train_err_pegasos = np.mean(train_err_pegasos, axis=1)
    avg_test_err_pegasos = np.mean(test_err_pegasos, axis=1)
    plt.figure()
    # color_list = ['b', 'r', 'g', 'c', ]
    # marker_list = ['o', 'x', '>', 's']

    plt.loglog(gamma_list, avg_train_err_libsvm, 'bo-', label='libsvm train')
    plt.loglog(gamma_list, avg_test_err_libsvm, 'ro-', label='libsvm test')
    plt.loglog(gamma_list, avg_train_err_dsvm, 'gx-', label='dsvm train')
    plt.loglog(gamma_list, avg_test_err_dsvm, 'cx-', label='dsvm test')
    plt.loglog(gamma_list, avg_train_err_pegasos, 'mD-', label='pegasos train')
    plt.loglog(gamma_list, avg_test_err_pegasos, 'kD-', label='pegasos test')
    plt.legend(bbox_to_anchor=(0, 1.17, 1, .1), loc=2, ncol=2, mode="expand", borderaxespad=0)
    plt.savefig('../output/usps_diff_gamma.pdf')
    # return avg_train_err_libsvm, avg_test_err_libsvm


if __name__ == "__main__":
# def comparison(x, y):
    perc = 0.8
    print '--------------------------------------------------------------------'
    print "usps dataset, size=%d, dim=%d, %2d%% for training" % (x.shape[0], x.shape[1], 100*(1-perc))
    if random_state is None:
        random_state = np.random.random_integers(low=0, high=1000)
    x_train, x_test, y_train, y_test = cv.train_test_split(x, y, test_size=perc, random_state=random_state)
    # scalar = preprocessing.StandardScaler().fit(x_train)
    # x_test = scalar.transform(x_test)
    # x_train = scalar.transform(x_train)
    mm_scale = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_train = mm_scale.fit_transform(x_train)
    x_test = mm_scale.transform(x_test)
    ntr = x_train.shape[0]
    # gm = 1.0/1
    gm = 1.0/ntr
    # lmda = 1/float(ntr)
    lmda = 1000/float(ntr)
    print('train dual svm')
    dsvm = DualKSVM(lmda=lmda, gm=gm, kernelstr='rbf', nsweep=0.8 * ntr, b=5, c=1)
    dsvm.fit(x_train, y_train, x_test, y_test, )

    print ('train Pegasos')
    kpega = Pegasos(lmda=lmda, gm=gm, kernelstr='rbf', nsweep=3)
    kpega.fit(x_train, y_train, x_test, y_test)

    clf = svm.SVC(C=lmda*ntr, kernel='rbf', gamma=gm, verbose=True,)
#    clf_da = svm.SVC(C=lmda*ntr, kernel='rbf', gamma=gm, fit_intercept=False)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    err_libsvm = zero_one_loss(pred, y_test)
    print "sklearn err %f" % err_libsvm

    plt.figure()
    plt.plot(dsvm.nker_opers, dsvm.err_tr, 'rx-', label='dc train error')
    plt.plot(kpega.nker_opers, kpega.err_tr, 'b.-', label='pegasos train error')
    plt.plot(dsvm.nker_opers, dsvm.err_te, 'gx-', label='dc test error')
    plt.plot(kpega.nker_opers, kpega.err_te, 'y.-', label='pegasos test error')
    plt.xlabel('number of kernel products')
    if one_vs_rest:
        plt.title('usps %d vs rest' % pos_class)
    else:
        plt.title('usps %d vs %d' % (pos_class, neg_class))
    plt.legend(loc='best')
    plt.savefig('../output/usps_%d_err.pdf' % pos_class)

    # plt.figure()
    # plt.plot(dsvm.nker_opers, dsvm.obj_primal, 'rx-', label='dc obj')
    # plt.plot(kpega.nker_opers, kpega.obj, 'b.-', label='pegasos obj')
    # plt.plot(dsvm.nker_opers, dsvm.obj, 'mx-', label='dc dual obj')
    # plt.ylim(-3, 10)
    # if one_vs_rest:
    #     plt.title('usps %d vs rest' % pos_class)
    # else:
    #     plt.title('usps %d vs %d' % (pos_class, neg_class))
    #
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig('../output/usps_%d_obj.pdf' % pos_class, format='pdf')

    # plt.figure()
    # plt.plot(dsvm.nker_opers, dsvm.nnzs, 'rx-', label='dc')
    # plt.plot(kpega.nker_opers, kpega.nnzs, 'b.-', label='pegasos')
    # plt.ylabel('number of non-zeros')
    # plt.legend(loc='best')
    # plt.savefig('../output/usps_%d_nnz.pdf' % pos_class, format='pdf')

    # plt.figure()
    # a = np.sort(dsvm.alpha)
    # plt.plot(a[::-1]/a.max(), 'rx-', label='dc')
    # b = np.sort(kpega.alpha)
    # plt.plot(b[::-1]/b.max(), 'b.-', label='pegasos')
    # plt.legend(loc='best')
    # plt.savefig('../output/usps_%d_weight.pdf' % pos_class, format='pdf')
    # plt.figure()
    # plt.plot(dsvm.nker_opers, dsvm.snorm_grad, 'r', label='grad snorm')
    # plt.legend()
    # return dsvm, kpega


def profile_usps(pos_class=3, neg_class=None, random_state=None):
    data = load_usps()
    if neg_class is None:
        x, y = convert_one_vs_all(data, pos_class)
        one_vs_rest = True
    else:
        x, y = convert_binary(data, pos_class, neg_class)
        one_vs_rest = False
    perc = 0.7
    print '--------------------------------------------------------------------'
    print "profile on usps dataset, size=%d, dim=%d, %2d%% for training" % (x.shape[0], x.shape[1], 100*perc)
    if random_state is None:
        random_state = np.random.random_integers(low=0, high=1000)
    x_train, x_test, y_train, y_test = cv.train_test_split(x, y, train_size=perc, test_size=round(1-perc,3), random_state=random_state)
    scalar = preprocessing.StandardScaler().fit(x_train)
    x_test = scalar.transform(x_test)
    x_train = scalar.transform(x_train)
    ntr = x_train.shape[0]
    gm = 1.0/ntr
    lmda = 100/float(ntr)
    dsvm = DualKSVM(ntr, lmda=lmda, gm=gm, kernelstr='rbf', nsweep=ntr/3, batchsize=5)
    dsvm.profile_scd_cy(x_train, y_train, x_test, y_test)


# if __name__ == '__main__':
#     profile_usps(8)
    # plot_convergence(8)
