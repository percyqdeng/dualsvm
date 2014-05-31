__author__ = 'qdengpercy'


from dualksvm import *
from stocksvm import *
from sklearn import svm
import sklearn.cross_validation as cv
import os
import time
import numpy as np
from load_data import *


if __name__ == '__main__':

    k = 4
    j = 1
    data = load_usps()
    x, y = convert_binary(data, k, j)
    print ' load usps dataset, n_samples=%d, dim=%d' % (x.shape[0], x.shape[1])
    n_rep = 1
    n_samples = x.shape[0]
    ss = cv.ShuffleSplit(n=n_samples, n_iter=n_rep, test_size=0.5, train_size=0.5, random_state=2)
    for tr_ind, te_ind in ss:
        xtr = x[tr_ind, :]
        ytr = y[tr_ind]
        xte = x[te_ind, :]
        yte = y[te_ind]
        ntr = xtr.shape[0]
        gm = 10
        dsvm = DualKSVM(ntr, lmda=1.0/ntr, gm=gm, kernel='rbf', nsweep=ntr/5, batchsize=5)
        dsvm.train_test(xtr, ytr, xte, yte, )

        kpega = Pegasos(ntr, lmda=1.0/ntr, gm=gm, kernel='rbf', nsweep=2)
        kpega.train_test(xtr, ytr, xte, yte)

    row = 1
    col = 2
    plt.subplot(row, col, 1)
    plt.semilogx(dsvm.n_ker_oper, dsvm.err_tr, 'r-', label='dc train error')
    plt.semilogx(kpega.n_ker_oper, kpega.err_tr, 'b-', label='pegasos train error')
    plt.semilogx(dsvm.n_ker_oper, dsvm.err_te, 'g-', label='dc test error')
    plt.semilogx(kpega.n_ker_oper, kpega.err_te, 'y-', label='pegasos test error')
    # plt.ylim(0, 0.07)
    plt.xlim(3000)
    plt.xlabel('number of kernerl product')
    plt.title('usps dataset')
    plt.legend(loc='best')
    plt.subplot(row, col, 2)
    # plt.figure()
    plt.semilogx(dsvm.n_ker_oper, dsvm.obj_primal, 'r-', label='dc obj')
    plt.semilogx(kpega.n_ker_oper, kpega.obj, 'b-', label='pegasos obj')
    plt.semilogx(dsvm.n_ker_oper, dsvm.obj, 'm-', label='dc dual obj')
    plt.ylim(-3, 10)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # plt.tight_layout()
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('../output/experi_usps.pdf', format='pdf')




