__author__ = 'qdpercy'

import sklearn.cross_validation as cv
import sklearn.preprocessing as preprocessing
from sklearn.metrics import zero_one_loss
from mysvm import *
import time


class Pegasos(MySVM):

    def __init__(self, lmda=0.01, gm=1, deg=0, kernelstr='rbf', verbose=True, nsweep=4, batchsize=1):
        super(Pegasos, self).__init__(lmda=lmda, gm=gm, deg=deg, kernelstr=kernelstr, nsweep=nsweep, b=5, c=1)
        self.batchsize = batchsize
        self.verbose = verbose

    def fit(self, xtr, ytr, xte=None, yte=None):
        self.xtr = xtr
        self.ytr = ytr
        self.ktr = self.kernel_matrix(xtr)
        if xte is None:
            kte = None
        else:
            kte = self.kernel_matrix(xte, xtr)
        self._fit(kte, yte)

    def predict(self, xte):
        kte = self.kernel_matrix(xte, self.xtr)
        if self.alpha is None:
            print('need to train svm first')
            exit(1)
        pred = np.sign(kte.dot(self.alpha * self.ytr)).astype(int)
        return pred

    def _fit(self, kte=None, yte=None):
        ntrain = self.ktr.shape[0]
        yktr = (self.ytr[:, np.newaxis] * self.ktr) * self.ytr[np.newaxis, :]
        alpha = np.zeros(ntrain)
        showtimes = 5
        rec_step = 1
        t = 1
        flag = np.zeros(ntrain)
        num_sv = 0
        count = 0
        num_hit = 0
        interval = self.nsweep * ntrain / 20
        scalar = 2
        num_iter = 1
        start = time.time()
        for iter in range(self.nsweep):
            perm = np.random.permutation(ntrain)
            for j in xrange(ntrain):
                i = perm[j]
                res = np.dot(yktr[i, :], alpha)/(self.lmda*t)
                count += num_sv
                if res < 1:
                    num_hit += 1
                    alpha[i] += 1
                    if flag[i] == 0:
                        flag[i] = 1
                        num_sv += 1
            # alpha /= self.lmda * self.T
                if self.verbose and t == num_iter:
                    self.nker_opers.append(count)
                    yka = np.dot(yktr, alpha/(self.lmda * t))
                    self.err_tr.append(np.mean(yka < 0))
                    obj = 1.0/ntrain * np.maximum(1-yka, 0).sum() + self.lmda/2*np.dot(alpha/(self.lmda * t), yka)
                    self.obj.append(obj)
                    self.nnzs.append(num_sv)
                    if kte is not None:
                        pred = np.sign(np.dot(kte, self.ytr*alpha/(self.lmda * t)))
                        self.err_te.append(np.mean(yte != pred))
                    # num_iter += interval
                    num_iter *= scalar
                t += 1
            # if k % (self.nsweep) == 0:
            # print "# of sweeps " + str(k)
        # print "num of hit %d" % num_hit
        # print "time cost %f " % (time.time()-start)
        self.alpha = alpha / (self.lmda * t)


    @staticmethod
    def tune_parameter(x,y, gmlist=[0.1], Clist=[1.0]):
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
                    clf = Pegasos(lmda=C/ntrain, gm=gm, kernelstr='rbf',  verbose=False )
                    clf.fit(xtrain, ytrain)
                    pred = clf.predict(xvalid)
                    err[i, j] += zero_one_loss(pred, yvalid)
        row, col = np.unravel_index(err.argmin(), err.shape)
        return gmlist[row], Clist[col]


if __name__ == "__main__":
    if os.name == "nt":
        dtpath = "..\\..\\dataset\\ucibenchmark\\"
    elif os.name == "posix":
        dtpath = '../../dataset/benchmark_uci/'
    filename = 'splicemat.mat'
    filename = 'bananamat'
    data = scipy.io.loadmat(dtpath + filename)
