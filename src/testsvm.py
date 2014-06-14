__author__ = 'qdengpercy'

from dualksvm import *
from stocksvm import *
from sklearn import svm
import os
import time
import numpy as np

class Test_SVM(object):
    """
    Test_SVM, using the uci benchmarkmat
    """
    def __init__(self, dtname='bananamat.mat'):
        if os.name == "nt":
            dtpath = "..\\..\\dataset\\ucibenchmark\\"
        elif os.name == "posix":
            dtpath = '../../dataset/benchmark_uci/'
        data = scipy.io.loadmat(dtpath + dtname)
        self.x = data['x']
        # self.x = preprocess_data(self.x)
        self.y = data['t']
        self.y = (np.squeeze(self.y)).astype(np.intc)

        self.train_ind = data['train'] - 1
        self.test_ind = data['test'] - 1
        print "use dataset: "+str(dtname)
        print "n = "+str(self.x.shape[0])+" p = "+str(self.x.shape[1])

    @staticmethod
    def _normalize_features(xtr, xte):
        std = np.std(xtr, 0)
        avg = np.mean(xtr, 0)
        xtr = (xtr - avg[np.newaxis, :]) / std[np.newaxis, :]
        xte = (xte - avg[np.newaxis, :]) / std[np.newaxis, :]
        return xtr, xte

    def _gen_i_th(self, i=-1):
        rep = self.train_ind.shape[0]
        if i < 0 or i >= rep:
            i = np.random.randint(low=0, high=rep)
        xtr = self.x[self.train_ind[i, :], :]
        ytr = self.y[self.train_ind[i, :]]
        xte = self.x[self.test_ind[i, :], :]
        yte = self.y[self.test_ind[i, :]]
        return xtr, ytr, xte, yte

    def run_profile(self):
        """
        get cprofile
        """
        xtr, ytr, xte, yte = self._gen_i_th(i=-1)
        ntr = xtr.shape[0]
        lmd = 10.0/ntr
        gamma = 1
        xtr, xte = Test_SVM._normalize_features(xtr, xte)

        start = time.time()
        # d1 = DualKSVM(n=ntr, lmda=lmd, gm=gamma, kernel='rbf', nsweep=2 * ntr, batchsize=10)

        # d1.train_test(xtr, ytr, xte, yte, algo_type="naive")
        # print "time 1 "+str(time.time() - start)
        # start = time.time()
        d2 = DualKSVM(n=ntr, lmda=lmd, gm=gamma, kernel='rbf', nsweep=2 * ntr, batchsize=1)
        d2.train_test(xtr, ytr, xte, yte, algo_type="cython")
        print "time 2 "+str(time.time() - start)
        return d2

    def rand_cmp_svm(self):
        xtr, ytr, xte, yte = self._gen_i_th(i=-1)
        ntr = xtr.shape[0]
        lmd = 10.0/ntr
        gamma = .1
        xtr, xte = Test_SVM._normalize_features(xtr, xte)
        # lmda=100 / float(ntr)

        dsvm = DualKSVM(n=ntr, lmda=lmd, gm=gamma, kernel='rbf', nsweep=20 * ntr, batchsize=1)
        start = time.time()
        dsvm.train_test(xtr, ytr, xte, yte, algo_type="cython")
        print "time 1 "+str(time.time() - start)
        kpega = Pegasos(n=ntr, lmda=lmd, gm=gamma, kernel='rbf', nsweep=20, batchsize=1)
        start = time.time()
        kpega.train_test(xtr, ytr, xte, yte)
        print "time 2 "+str(time.time() - start)
        plt.subplot(2, 2, 1)
        plt.plot(dsvm.ker_oper, dsvm.err_tr, 'rx-', label="dualcoor")
        plt.plot(kpega.ker_oper, kpega.err_tr, 'bo-', label="pegasos")
        plt.legend(loc="best")
        plt.ylabel("train err")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # plt.figure()
        plt.subplot(2, 2, 2)
        plt.plot(dsvm.ker_oper, dsvm.err_te, 'rx-', label="dualcoor")
        plt.plot(kpega.ker_oper, kpega.err_te, 'bo-', label="pegasos")
        plt.legend(loc="best")
        plt.ylabel("test err")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()

        # plt.figure()
        plt.subplot(2,2,3)
        plt.plot(dsvm.ker_oper, (-np.asarray(dsvm.obj)), 'rx-', label="sdc dual obj")
        plt.plot(dsvm.ker_oper, (np.asarray(dsvm.obj_primal)), 'gx-', label="sdc primal obj")
        # plt.subplot(122)
        plt.plot(kpega.ker_oper, (kpega.obj), 'bo-', label="pegasos")
        plt.legend(loc='best')
        plt.ylabel("obj")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

        return dsvm, kpega

def test_benchmark(data):
    x = data['x']
    y = data['t']
    y = np.ravel(y)
    trInd = data['train'] - 1
    teInd = data['test'] - 1
    rep = trInd.shape[0]
    rep = 1
    libsvm_err = np.zeros(rep)
    for i in range(rep):
        if i % 10 == 0:
            print "iteration #: " + str(i)
        ntr = len(y[trInd[i, :]])
        xtr = x[trInd[i, :], :]
        ytr = y[trInd[i, :]]
        dsvm = DualKSVM(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', nsweep=20, batchsize=5)
        dsvm.train(xtr, ytr)
        clf = svm.SVC(kernel='rbf')
        clf.fit(xtr, ytr)
        pred = clf.predict(x[teInd[i, :], :])
        # pred = np.ravel(pred)
        libsvm_err[i] = np.mean(pred != y[teInd[i, :]])
    dsvm.plot_train_result()
    return libsvm_err


if __name__ == "__main__":
    os.system("%load_ext autoreload")
    if os.name == "nt":
        dtpath = "..\\..\\dataset\\ucibenchmark\\"
    elif os.name == "posix":
        dtpath = '../../dataset/benchmark_uci/'
    filename = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
    # dsvm = test_dualsvm(data)
    newtest = Test_SVM(filename[1])
    # dsvm, kpega = newtest.rand_cmp_svm()
    # newtest.run_profile()