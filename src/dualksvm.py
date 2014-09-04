import os
import sys
import scipy.io
import sklearn.cross_validation as cv
import sklearn.preprocessing as preprocessing
from sklearn.metrics import zero_one_loss
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from mysvm import *
from coor import *
import scg_svm

# pyximport.install()


class DualKSVM(MySVM):
    """Dual randomized stochastic coordinate descent for kenel method, SVM and ridge regression

    """

    def __init__(self, lmda=0.01, gm=1.0, kernel='rbf', rho=0.1, nsweep=None, verbose=True, b=4, c=1, algo_type='scg_da'):
        """
        :param algo_type: scg_md, scg mirror descent; cd, coordinate descent; scg_da, scg dual average
        """
        super(DualKSVM, self).__init__(lmda, gm, kernel, nsweep, b=b, c=c)
        # C = 1.0 / n  # box constraint on the dual
        self.obj_primal = []
        self.rho = rho
        self.algo_type = algo_type
        self.verbose = verbose

    def fit(self, xtr, ytr, xte=None, yte=None):
        self.set_train_kernel(xtr)
        self.xtr = xtr
        self.ytr = ytr
        self.has_kte = not(xte is None)
        if self.nsweep is None:
            if self.algo_type =='scg_da':
               self.nsweep = xtr.shape[0]
            elif self.algo_type == 'cd':
                self.nsweep = xtr.shape[0]


        if not self.has_kte:
            # if self.algo_type == 'scg_md':
            #     self._stoc_coor_cython()
            if self.algo_type == 'cd':
                self.alpha, self.err_tr, self.obj, self.nker_opers = \
                    scg_svm.cd_svm(ktr=self.ktr, ytr=self.ytr, verbose=self.verbose, lmda=self.lmda, nsweep=np.int(self.nsweep))
            elif self.algo_type == 'scg_da':
                self.alpha, self.err_tr, self.obj, self.obj_primal, self.nker_opers, self.nnzs, = \
                    scg_svm.scg_da_svm(ktr=self.ktr, ytr=self.ytr, lmda=self.lmda, rho=self.rho, verbose=self.verbose,
                                       nsweep=np.int(self.nsweep), b=np.int(self.b), c=np.int(self.c))
        else:
            self.set_test_kernel(xtr, xte)
            self.yte = yte

            # if self.algo_type == 'scg_md':
            #     self._stoc_coor_cython()
            if self.algo_type == 'cd':
                self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers = \
                    scg_svm.cd_svm(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=yte, verbose=self.verbose, lmda=self.lmda, nsweep=np.int(self.nsweep))
            elif self.algo_type == 'scg_da':
                self.alpha, self.err_tr, self.err_te, self.obj, self.obj_primal, self.nker_opers, self.nnzs, = \
                    scg_svm.scg_da_svm(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=yte, lmda=self.lmda, rho=self.rho,
                                       verbose=self.verbose, nsweep=np.int(self.nsweep), b=np.int(self.b), c=np.int(self.c))

    def predict(self, xte):
        self.set_test_kernel(self.xtr, xte)
        if self.alpha is None:
            print('need to train svm first')
            exit(1)
        pred = np.sign(self.kte.dot(self.alpha * self.ytr)).astype(int)
        return pred

    def profile_scd_cy(self, xtr, ytr, xte, yte):
        import pstats
        import cProfile
        import pyximport

        pyximport.install()
        self.set_train_kernel(xtr)
        self.set_test_kernel(xtr, xte)
        self.has_kte = True
        self.ytr = ytr
        T = self.nsweep * self.n - 1
        str = "coor_cy.scd_cy(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=self.yte, lmda=self.lmda,\
                       nsweep=np.int(self.nsweep), T=int(T), batchsize=np.int(self.batchsize))"
        cProfile.runctx(str, globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()

    @staticmethod
    def tune_parameter(x,y, gmlist=[0.1], Clist=[1.0], algo_type='scg_da'):
        # cross validation to tweak the parameter
        n, p = x.shape
        err = np.zeros((len(gmlist), len(Clist)))
        kf = cv.KFold(n, n_folds=3)
        for train_ind, valid_ind in kf:
            xtrain = x[train_ind, :]
            ytrain = y[train_ind]
            ntrain = ytrain.size
            xvalid = x[valid_ind, :]
            yvalid = y[valid_ind]
            for i, gm in enumerate(gmlist):
                for j, C in enumerate(Clist):
                    clf = DualKSVM(lmda=C/ntrain, gm=gm, kernel='rbf', rho=0.1, b=5, c=1, verbose=False, algo_type=algo_type)
                    clf.fit(xtrain, ytrain)
                    pred = clf.predict(xvalid)
                    err[i, j] += zero_one_loss(pred, yvalid)
        row, col = np.unravel_index(err.argmin(), err.shape)
        return gmlist[row], Clist[col]

def test_dualsvm(data):
    x = data['x']
    y = data['t']
    y = np.ravel(y)
    trInd = data['train'] - 1
    teInd = data['test'] - 1
    i = np.random.choice(trInd.shape[0])
    # i = 10
    ntr = len(y[trInd[i, :]])
    xtr = x[trInd[i, :], :]
    ytr = y[trInd[i, :]]
    xte = x[teInd[i, :], :]
    yte = y[teInd[i, :]]
    dsvm = DualKSVM(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', nsweep=1000, batchsize=20)
    dsvm.train_test(xtr, ytr, xte, yte)
    dsvm.plot_train_result()
    return dsvm


if __name__ == "__main__":
    if os.name == "nt":
        dtpath = "..\\..\\dataset\\ucibenchmark\\"
    elif os.name == "posix":
        dtpath = '../../dataset/benchmark_uci/'
    filename = ["bananamat", "breast_cancermat", "cvt_bench", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
    data = scipy.io.loadmat(dtpath + filename[-3])
    # err = test_benchmark(data)
    dsvm = test_dualsvm(data)
