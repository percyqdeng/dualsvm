import scipy.io
import sklearn.cross_validation as cv
from sklearn.metrics import zero_one_loss
from kernel_func import *
from mysvm import *
import dsvm
# import scg_svm_on_fly


class DualKSVM(MySVM):
    """Dual randomized stochastic coordinate descent for kenel method, SVM and ridge regression

    """

    def __init__(self, lmda=0.01, gm=1.0, deg=0, kernelstr='rbf', nsweep=None, b=4, c=1, rho=0.1, verbose=True,
                 algo_type='scg_da'):
        """
        :param algo_type: scg_md, scg mirror descent; cd, coordinate descent; scg_da, scg dual average
        """
        super(DualKSVM, self).__init__(lmda, gm, deg, kernelstr, nsweep, b, c)
        # C = 1.0 / n  # box constraint on the dual
        self.rho = rho
        self.algo_type = algo_type
        self.verbose = verbose

    def fit(self, xtr, ytr, xte=None, yte=None):
        n = xtr.shape[0]
        self.xtr = xtr
        self.ytr = ytr
        if n < 1000:
            self._fit_precom(xte, yte)
        else:
            self._fit_on_fly(xte, yte)

    def _fit_precom(self, xte, yte):
        # ---------------precompute the kernel-----------------
        self.ktr = self.kernel_matrix(self.xtr)
        if xte is not None:
            self.kte = self.kernel_matrix(xte, self.xtr)
        if self.nsweep is None:
            if self.algo_type == 'scg_da':
                self.nsweep = self.xtr.shape[0]
            elif self.algo_type == 'cd':
                self.nsweep = self.xtr.shape[0]

        if self.algo_type == 'cd':
            res = dsvm.coord_descent(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=yte, verbose=self.verbose,
                                 lmda=self.lmda, nsweep=np.int(self.nsweep))
        elif self.algo_type == 'scg_da':
            res = dsvm.stoc_dual_averaging(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=yte, lmda=self.lmda, rho=self.rho,
                                      verbose=self.verbose, nsweep=np.int(self.nsweep), b=np.int(self.b), c=np.int(self.c))
        self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers = res

    def _fit_on_fly(self, xte, yte):
        if self.algo_type == 'cd':
            res = dsvm.coord_descent(xtr=self.xtr, ytr=self.ytr, kernel=self.kernel, xte=xte, yte=yte,
                                             verbose=self.verbose, lmda=self.lmda, nsweep=np.int(self.nsweep))
        elif self.algo_type == 'scg_da':
            res = dsvm.stocda_on_fly(xtr=self.xtr, ytr=self.ytr, kernel=self.kernel, xte=xte, yte=yte,
                                                lmda=self.lmda, rho=self.rho, verbose=np.int(self.verbose),
                                                nsweep=np.int(self.nsweep), b=np.int(self.b), c=np.int(self.c))
        self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers = res

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
        self.ytr = ytr
        T = self.nsweep * self.n - 1
        str = "coor_cy.scd_cy(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=self.yte, lmda=self.lmda,\
                       nsweep=np.int(self.nsweep), T=int(T), batchsize=np.int(self.batchsize))"
        cProfile.runctx(str, globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()

        @staticmethod
        def tune_parameter(x, y, gmlist=[0.1], Clist=[1.0], algo_type='scg_da'):
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
                        clf = DualKSVM(lmda=C / ntrain, gm=gm, kernelstr='rbf', rho=0.1, b=5, c=1, verbose=False,
                                       algo_type=algo_type)
                        clf.fit(xtrain, ytrain)
                        pred = clf.predict(xvalid)
                        err[i, j] += zero_one_loss(pred, yvalid)
            row, col = np.unravel_index(err.argmin(), err.shape)
            return gmlist[row], Clist[col]
