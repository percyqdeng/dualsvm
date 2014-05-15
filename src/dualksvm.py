__author__ = 'qdengpercy'

import scipy.io
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import numpy.linalg as la


class DualKSVM:
    """Dual randomized stochastic coordinate descent for kenel method, SVM and ridge regression

    lmda: int, regularizer
    alpha:  array type, weights of kernel classifier
    T : int, maximal number of iteration - 1
    C : anoter regularizer, essentially C=1/lmda*n

    kernel : string type,
        "rbf" : gaussian kernel
        "poly" : polynomial kernel
    gm : parameter in rbf kernel

    err_tr
    """

    def __init__(self, n, lmda=0.01, gm=1, kernel='rbf', nsweep=1000, batchsize=2):
        self.batchsize = batchsize
        self.alpha = np.zeros(n)
        self.lmda = lmda
        self.c = 1.0 / (n * lmda)
        self.dim = n
        self.gm = gm
        self.nsweep = nsweep
        self.T = nsweep * n - 1
        self.kernel = kernel
        self.obj = []
        self.nnz = []
        self.err_tr = []

    def train(self, x, y):
        K = self._set_kernels(x)
        self.rand_stoc_coor(y, K)

    def test(self, x, y):

        pass

    def rand_stoc_coor(self, y, K):
        """
        stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
        variables
        """
        n = K.shape[0]
        yky = (y[:, np.newaxis] * K) * y[np.newaxis, :]
        lip = np.diag(yky)
        L_t = np.max(lip)
        Q = 1
        D_t = Q * self.dim * self.c ** 2 / 2
        sig_list = self._esti_std(yky)
        sig = np.sqrt((sig_list ** 2).sum())

        eta = np.ones(self.T + 1)
        eta *= np.minimum(1.0 / (2 * L_t), D_t / sig * np.sqrt(float(self.dim) / (1 + self.T)))
        theta = eta + .0
        alpha = np.zeros(n)
        a_tilde = np.zeros(n)
        delta = np.zeros(self.T + 2)
        uu = np.zeros(n, dtype=int)
        # index of the active iteration for each coordinate, u[i] = t means the most recent update of
        # ith coordinate is in the t-th round, t = 0,1,...,T
        # print "gamma: " + str(gamma) + " eta: " + str(eta)
        showtimes = 5
        t = 0
        for i in range(self.nsweep):
            # index of batch data to compute stochastic coordinate gradient
            samp = np.random.choice(n, size=(n, self.batchsize))
            samp = range(n)
            # index of sampled coordinate to update
            perm = np.random.permutation(n)
            for j in range(n):
                # samp_ind = samp[j, :]
                samp_ind = samp
                var_ind = perm[j]
                delta[t + 1] = delta[t] + theta[t]
                subk = yky[var_ind, samp_ind]
                stoc_coor_grad = np.dot(subk, alpha[samp_ind]) * float(n) / self.batchsize - 1
                a_tilde[var_ind] += (delta[t + 1] - delta[uu[var_ind]]) * alpha[j]
                alpha[var_ind] = self._prox_mapping(g=stoc_coor_grad, x0=alpha[var_ind], r=eta[t])
                assert(all(0 <= x <= self.c for x in np.nditer(alpha[var_ind])))  #only works for size 1
                uu[var_ind] = t + 1
                t += 1
            if i % (self.nsweep / showtimes) == 0:
                print "# of sweeps " + str(i)

            a_avg = a_tilde / delta[t]
            assert(all(0 <= x <= self.c for x in np.nditer(a_avg)))
            yha = np.dot(yky, a_avg)
            res = self.lmda * (0.5 * np.dot(a_avg, yha) - a_avg.sum())
            self.obj.append(res)
            nnzs = (a_avg != 0).sum()
            self.nnz.append(nnzs)
            err = np.mean(yha < 0)
            self.err_tr.append(err)
        # averaging
        a_tilde += (delta[self.T + 1] - delta[uu]) * (self.T - uu + 1) * alpha
        self.alpha = a_tilde / delta[self.T + 1]

    def plot_train_result(self):
        row = 2
        col = 2
        plt.subplot(row, col, 1)
        plt.plot(self.obj, 'b-', label="objective")
        plt.subplot(row, col, 2)
        plt.plot(self.nnz, 'b-', label="# of nnzs")
        plt.subplot(row, col, 3)
        plt.plot(self.err_tr, label="training error")

    def _set_kernels(self, x):
        if self.kernel == 'rbf':
            std = np.std(x, axis=0)
            x = x / std[np.newaxis, :]
            xsquare = np.sum(x ** 2, 1)
            xxT = np.dot(x, x.T)
            dist = xsquare[:, np.newaxis] - 2 * xxT + xsquare[np.newaxis, :]
            K = np.exp(-self.gm * dist)
        else:
            print "the other kernel tbd"
        return K

    def _prox_mapping(self, g, x0, r):
        """
        proximal coordinate gradient mapping
        argmin  x*g + 1/gamma*D(x0,x)
        """
        x = x0 - r * g
        x = np.minimum(np.maximum(0, x), self.c)

        return x

    def _esti_std(self, K):
        n = K.shape[0]
        sig = np.zeros(self.dim)
        alpha = np.random.uniform(0, self.c, self.dim)
        rep = 100
        for i in range(self.dim):
            g = K[i, :] * alpha
            sig[i] = np.std(g) * n / np.sqrt(self.batchsize)

        return sig


def test_dualsvm(data):
    x = data['x']
    y = data['t']
    y = np.ravel(y)
    trInd = data['train'] - 1
    teInd = data['test'] - 1
    i = np.random.choice(trInd.shape[0])
    ntr = len(y[trInd[i, :]])
    xtr = x[trInd[i, :], :]
    ytr = y[trInd[i, :]]
    dsvm = DualKSVM(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', nsweep=1000, batchsize=10)
    dsvm.train(xtr, ytr)
    dsvm.plot_train_result()
    return dsvm


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
    datapath = "/Users/qdengpercy/workspace/dataset/benchmark_uci/"
    filename = 'bananamat.mat'
    data = scipy.io.loadmat(datapath + filename)
    # err = test_benchmark(data)
    dsvm = test_dualsvm(data)