__author__ = 'qdengpercy'


from dualksvm import *
from stocksvm import *
from sklearn import svm
import sklearn.cross_validation as cv
import os
import time
import numpy as np
from load_data import *


def plot_convergence(pos_class=3, neg_class=8, random_state=None):
    """
    plot of convergence for usps data on binary classification
    :param pos_class: the digit as positive class
    :param neg_class: the digit as negative class
    :return:
    """
    data = load_usps()
    x, y = convert_binary(data, pos_class, neg_class)
    print ' load usps dataset, n_samples=%d, dim=%d' % (x.shape[0], x.shape[1])
    if random_state is None:
        random_state = 2
    xtr, xte, ytr, yte = cv.train_test_split(x, y, train_size=0.5, test_size=0.5, random_state=random_state)
    ntr = xtr.shape[0]
    gm = 10
    dsvm = DualKSVM(ntr, lmda=1.0/ntr, gm=gm, kernel='rbf', nsweep=ntr/5, batchsize=5)
    dsvm.train_test(xtr, ytr, xte, yte, )

    kpega = Pegasos(ntr, lmda=1.0/ntr, gm=gm, kernel='rbf', nsweep=2)
    kpega.train_test(xtr, ytr, xte, yte)

    row = 1
    col = 2
    # plt.subplot(row, col, 1)
    plt.figure()

    plt.loglog(dsvm.n_ker_oper, dsvm.err_tr, 'rx-', label='dc train error')
    plt.loglog(kpega.n_ker_oper, kpega.err_tr, 'b.-', label='pegasos train error')
    plt.loglog(dsvm.n_ker_oper, dsvm.err_te, 'gx-', label='dc test error')
    plt.loglog(kpega.n_ker_oper, kpega.err_te, 'y.-', label='pegasos test error')
    # plt.ylim(0, 0.07)
    # plt.xlim(3000)
    plt.xlabel('number of kernerl product')
    plt.title('usps %d vs %d' % (pos_class, neg_class))
    plt.legend(loc='best')
    plt.savefig('../output/usps_err.pdf')
    # plt.subplot(row, col, 2)
    plt.figure()
    plt.semilogx(dsvm.n_ker_oper, dsvm.obj_primal, 'r-', label='dc obj')
    plt.semilogx(kpega.n_ker_oper, kpega.obj, 'b-', label='pegasos obj')
    plt.semilogx(dsvm.n_ker_oper, dsvm.obj, 'm-', label='dc dual obj')
    plt.ylim(-3, 10)
    plt.title('usps %d vs %d' % (pos_class, neg_class))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # plt.tight_layout()
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('../output/usps_.pdf', format='pdf')


if __name__ == '__main__':
    plot_convergence(2, 8)
