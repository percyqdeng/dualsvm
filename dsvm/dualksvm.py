import scipy.io
import sklearn.cross_validation as cv
from sklearn.metrics import zero_one_loss
from kernel_func import *
from mysvm import *
from rand_dataset import RandomDataset
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
        self.xtr = xtr
        self.ytr = ytr
        self.construct_dataset(xtr, ytr, xte, yte)
        if self.algo_type == 'cd':
            res = dsvm.coord_descent(self.dataset, nsweep=np.int(self.nsweep), lmda=self.lmda, verbose=self.verbose)
            self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers = res
        elif self.algo_type == 'scg_da':
            res = dsvm.coord_dual_averaging(self.dataset, verbose=self.verbose, lmda=self.lmda, b=int(self.b),
                                            c=int(self.c), nsweep=np.int(self.nsweep), rho=self.rho)
            self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers = res
        elif self.algo_type == 'sbmd':
            res = dsvm.coord_mirror_descent(self.dataset, verbose=self.verbose, lmda=self.lmda, b=int(self.b),
                                            c=int(self.c), nsweep=np.int(self.nsweep), rho=self.rho)
            self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers, self.err_tr2, self.obj2= res
        else:
            raise NotImplementedError
        # self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers = res

    def predict(self, xte):
        self.kernel_matrix(xte, self.xtr)
        if self.alpha is None:
            print('need to train svm first')
            exit(1)
        pred = np.sign(self.kte.dot(self.alpha * self.ytr)).astype(int)
        return pred


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
