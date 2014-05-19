__author__ = 'qdengpercy'

import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from mysvm import *


class DualKSVM(MySVM):
    """Dual randomized stochastic coordinate descent for kenel method, SVM and ridge regression

    lmda: int, regularizer
    alpha:  array type, weights of kernel classifier
    T : int, maximal number of iteration - 1

    kernel : string type,
        "rbf" : gaussian kernel
        "poly" : polynomial kernel
    gm : parameter in rbf kernel
    err_tr
    """

    def __init__(self, n, lmda=0.01,  gm=1, kernel='rbf', nsweep=1000, batchsize=2):
        super(DualKSVM, self).__init__(n, lmda,  gm, kernel, nsweep, batchsize)
        self._cc = 1.0 / n   # box constraint on the dual

    def train(self, xtr, ytr):
        self.set_train_kernel(xtr)
        self.ytr = ytr
        self._rand_stoc_coor()

    def train_test(self, xtr, ytr, xte, yte):
        self.set_train_kernel(xtr)
        self.set_test_kernel(xtr, xte)
        self.has_kte = True
        self.ytr = ytr
        self.yte = yte
        self._rand_stoc_coor()

    def test(self, x, y):
        pass

    def _rand_stoc_coor(self):
        """
        stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
        variables
        """
        n = self.ktr.shape[0]
        yktr = (self.ytr[:, np.newaxis] * self.ktr) * self.ytr[np.newaxis, :]
        print"------------estimate parameters and set up variables----------------- "
        # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
        #
        lip = np.diag(yktr)/self.lmda
        l_max = np.max(lip)
        Q = 1
        D_t = Q * np.sqrt(1.0/(2*n))
        # D_t = Q * (n / 2) * self._cc
        sig_list = self._esti_std(yktr)
        sig = np.sqrt((sig_list ** 2).sum())
        eta = np.ones(self.T + 1)
        eta *= np.minimum(1.0 / (2 * l_max), D_t / sig * np.sqrt(float(self.num) / (1 + self.T)))
        theta = eta + .0
        alpha = np.zeros(n)  # the most recent solution
        a_tilde = np.zeros(n)  # the accumulated solution in the path
        delta = np.zeros(self.T + 2)
        uu = np.zeros(n, dtype=int)
        # index of update, u[i] = t means the most recent update of
        # ith coordinate is in the t-th round, t = 0,1,...,T
        showtimes = 5
        t = 0
        count = 0
        print "estimated sigma: "+str(sig)+" lipschitz: "+str(l_max)
        print "----------------------start the algorithm----------------------"
        for i in range(self.nsweep):
            # index of batch data to compute stochastic coordinate gradient
            samp = np.random.choice(n, size=(n, self.batchsize))
            # samp = np.random.permutation(n)
            # index of sampled coordinate to update

            perm = np.random.permutation(n)
            for j in range(n):
                samp_ind = samp[j, :]
                # samp_ind = samp[j]
                var_ind = perm[j]
                # var_ind = samp_ind
                delta[t + 1] = delta[t] + theta[t]
                subk = yktr[var_ind, samp_ind]
                # stoc_coor_grad = np.dot(subk, alpha[samp_ind]) * float(n) / self.batchsize - 1
                stoc_coor_grad = 1/self.lmda*(np.dot(subk, alpha[samp_ind]) * float(n) / self.batchsize) - 1
                a_tilde[var_ind] += (delta[t + 1] - delta[uu[var_ind]]) * alpha[var_ind]
                res = alpha[var_ind] - eta[t]*stoc_coor_grad
                if res < 0:
                    alpha[var_ind] = 0
                elif res <= self._cc:
                    alpha[var_ind] = res
                else:
                    alpha[var_ind] = self._cc
                # alpha[var_ind] = np.minimum(np.maximum(0, alpha[var_ind] - eta[t]*stoc_coor_grad), self._cc)
                # alpha[var_ind] = self._prox_mapping(g=stoc_coor_grad, x0=alpha[var_ind], r=eta[t])
                # assert(all(0 <= x <= self._cc for x in np.nditer(alpha[var_ind])))  #only works for size 1
                uu[var_ind] = t + 1
                t += 1
                count += self.batchsize
            if i % (self.nsweep / showtimes) == 0:
                print "# of sweeps " + str(i)
            #-------------compute the result after the ith sweep----------------
            a_avg = a_tilde + (delta[t]-delta[uu]) * alpha
            a_avg /= delta[t]
            # a_avg = alpha
            # assert(all(0 <= x <= self._cc for x in np.nditer(a_avg)))
            yka = np.dot(yktr, a_avg)
            res = 1.0/self.lmda * 0.5 * np.dot(a_avg, yka) - a_avg.sum()
            self.obj.append(res)
            # if i > 2 and self.obj[-1] > self.obj[-2]:
            #     print "warning"
            nnzs = (a_avg != 0).sum()
            self.nnz.append(nnzs)
            err = np.mean(yka < 0)
            self.err_tr.append(err)
            self.ker_oper.append(count)
            if self.has_kte:
                pred = np.sign(np.dot(self.kte, self.ytr*a_avg))
                err = np.mean(self.yte != pred)
                self.err_te.append(err)
        # -------------compute the final result after nsweep-th sweeps---------------
        a_tilde += (delta[self.T + 1] - delta[uu]) * alpha
        self.alpha = a_tilde / delta[self.T + 1]
        self.final = self.lmda * (0.5 * np.dot(self.alpha, np.dot(yktr, self.alpha)) - self.alpha.sum())
        self.bound1 = (n-1)*0.5*l_max/self.lmda
        self.bound2 = l_max
        self.bound3 = sig * np.sqrt(2)

    def plot_train_result(self):
        row = 1
        col = 2
        plt.figure()
        # plt.subplot(row, col, 1)
        plt.plot(self.obj, 'b-', label="stoc")
        seq = range(self.num, self.T+2, self.num)
        # bound = (self.bound1+self.bound2)/seq + self.bound3/np.sqrt(seq)
        # plt.plot((bound), 'r-', label="bound")
        plt.ylabel("obj")
        plt.legend()
        # plt.subplot(row, col, 2)
        plt.figure()
        plt.plot(self.err_tr)
        plt.ylabel("training error")
        # plt.subplot(row, col, 3)
        # plt.plot(self.nnz, 'b-', label="# of nnzs")
        plt.figure()
        plt.plot(self.ker_oper, self.err_te, 'r')

    def _prox_mapping(self, g, x0, r):
        """
        proximal coordinate gradient mapping
        argmin  x*g + 1/r*D(x0,x)
        """
        x = x0 - r * g
        x = np.minimum(np.maximum(0, x), 1.0/self.num)
        return x

    def _esti_std(self, kk):
        """
        estimate standard deviation of coordiante stochastic gradient
        """
        n = kk.shape[0]
        sig = np.zeros(self.num)
        alpha = np.random.uniform(0, self._cc, self.num)
        rep = 100
        for i in range(self.num):
            g = kk[i, :]/self.lmda * self._cc
            sig[i] = np.std(g) * n / np.sqrt(self.batchsize)
        return sig


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