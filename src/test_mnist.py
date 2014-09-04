
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
    plot of convergence for mnist data on binary classification
    :param pos_class: the digit as positive class
    :param neg_class: the digit as negative class
    :return:
    """
    data = load_mnist()
    x, y = convert_binary(data, pos_class, neg_class)
    x, y = convert_one_vs_all(data, pos_class)
    perc = 0.7
    print '--------------------------------------------------------------------'
    print "mnist dataset, size=%d, dim=%d, %f%% for training" % (x.shape[0], x.shape[1], 100*perc)
    if random_state is None:
        random_state = np.random.random_integers(low=0, high=1000)
    xtr, xte, ytr, yte = cv.train_test_split(x, y, train_size=perc, test_size=round(1-perc,3), random_state=random_state)
    ntr = xtr.shape[0]
    gm = 10
    dsvm = DualKSVM(lmda=1.0/ntr, gm=gm, kernel='rbf', nsweep=ntr/2, algo_type='scg')

    dsvm.fit(xtr, ytr, xte, yte,)
    kpega = Pegasos(lmda=1.0/ntr, gm=gm, kernel='rbf', nsweep=4)
    kpega.fit(xtr, ytr, xte, yte)
    plt.figure()

    plt.plot(dsvm.nker_opers, dsvm.err_tr, 'rx-', label='dc train error')
    plt.plot(kpega.nker_opers, kpega.err_tr, 'b.-', label='pegasos train error')
    plt.plot(dsvm.nker_opers, dsvm.err_te, 'gx-', label='dc test error')
    plt.plot(kpega.nker_opers, kpega.err_te, 'y.-', label='pegasos test error')
    # plt.ylim(0, 0.07)
    # plt.xlim(3000)
    plt.xlabel('number of kernerl operation')
    plt.title('mnist %d vs %d' % (pos_class, neg_class))
    plt.legend(loc='best')
    plt.savefig('../output/mnist_err.pdf')

    plt.figure()
    plt.plot(dsvm.nker_opers, dsvm.obj_primal, 'r-', label='dc obj')
    plt.plot(kpega.nker_opers, kpega.obj, 'b-', label='pegasos obj')
    plt.plot(dsvm.nker_opers, dsvm.obj, 'm-', label='dc dual obj')
    plt.ylim(-3, 10)
    plt.title('mnist %d vs %d' % (pos_class, neg_class))

    plt.legend(loc='best')
    # plt.show()
    plt.savefig('../output/mnist_.pdf', format='pdf')


if __name__ == '__main__':
    plot_convergence(7, 8)
