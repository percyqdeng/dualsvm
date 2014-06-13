
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn import preprocessing
import os
import time
import numpy as np

from load_data import *
from dualksvm import *
from stocksvm import *


def plot_convergence(pos_class=3, neg_class=None, random_state=None):
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
    print "usps dataset, size=%d, dim=%d, %2d%% for training" % (x.shape[0], x.shape[1], 100*perc)
    if random_state is None:
        random_state = np.random.random_integers(low=0, high=1000)
    xtr, xte, ytr, yte = cv.train_test_split(x, y, train_size=perc, test_size=round(1-perc,3), random_state=random_state)
    scalar = preprocessing.StandardScaler().fit(xtr)
    xte = scalar.transform(xte)
    xtr = scalar.transform(xtr)
    ntr = xtr.shape[0]
    gm = 1.0/ntr
    lmda = 100/float(ntr)
    dsvm = DualKSVM(ntr, lmda=lmda, gm=gm, kernel='rbf', nsweep=ntr/3, batchsize=2)
    dsvm.train_test(xtr, ytr, xte, yte, )

    kpega = Pegasos(ntr, lmda=lmda, gm=gm, kernel='rbf', nsweep=3)
    kpega.train_test(xtr, ytr, xte, yte)

    plt.figure()
    plt.loglog(dsvm.nker_opers, dsvm.err_tr, 'rx-', label='dc train error')
    plt.loglog(kpega.nker_opers, kpega.err_tr, 'b.-', label='pegasos train error')
    plt.loglog(dsvm.nker_opers, dsvm.err_te, 'gx-', label='dc test error')
    plt.loglog(kpega.nker_opers, kpega.err_te, 'y.-', label='pegasos test error')
    plt.xlabel('number of kernel product')
    if one_vs_rest:
        plt.title('usps %d vs rest' % pos_class)
    else:
        plt.title('usps %d vs %d' % (pos_class, neg_class))
    plt.legend(loc='best')
    plt.savefig('../output/usps_%d_err.pdf' % pos_class)

    plt.figure()
    plt.semilogx(dsvm.nker_opers, dsvm.obj_primal, 'rx-', label='dc obj')
    plt.semilogx(kpega.nker_opers, kpega.obj, 'b.-', label='pegasos obj')
    plt.semilogx(dsvm.nker_opers, dsvm.obj, 'mx-', label='dc dual obj')
    plt.ylim(-3, 10)
    if one_vs_rest:
        plt.title('usps %d vs rest' % pos_class)
    else:
        plt.title('usps %d vs %d' % (pos_class, neg_class))

    plt.legend(loc='best')
    # plt.show()
    plt.savefig('../output/usps_%d_obj.pdf' % pos_class, format='pdf')

    plt.figure()
    plt.semilogx(dsvm.nker_opers, dsvm.nnzs, 'rx-', label='dc')
    plt.semilogx(kpega.nker_opers, kpega.nnzs, 'b.-', label='pegasos')
    plt.ylabel('number of non-zeros')
    plt.legend(loc='best')
    plt.savefig('../output/usps_%d_nnz.pdf' % pos_class, format='pdf')

    plt.figure()
    a = np.sort(dsvm.alpha)
    plt.plot(a[::-1]/a.max(), 'rx-', label='dc')
    b = np.sort(kpega.alpha)
    plt.plot(b[::-1]/b.max(), 'b.-', label='pegasos')
    plt.legend(loc='best')
    plt.savefig('../output/usps_%d_weight.pdf' % pos_class, format='pdf')
    return dsvm, kpega


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
    xtr, xte, ytr, yte = cv.train_test_split(x, y, train_size=perc, test_size=round(1-perc,3), random_state=random_state)
    scalar = preprocessing.StandardScaler().fit(xtr)
    xte = scalar.transform(xte)
    xtr = scalar.transform(xtr)
    ntr = xtr.shape[0]
    gm = 1.0/ntr
    lmda = 100/float(ntr)
    dsvm = DualKSVM(ntr, lmda=lmda, gm=gm, kernel='rbf', nsweep=ntr/3, batchsize=5)
    dsvm.profile_scd_cy(xtr, ytr, xte, yte)


if __name__ == '__main__':
    # profile_usps(8)
    plot_convergence(8)