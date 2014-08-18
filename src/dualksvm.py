import os
import sys
import scipy.io
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

    def __init__(self, lmda=0.01, gm=1, kernel='rbf', rho=0.01, nsweep=None, b=4, c=1, algo_type='scg_md'):
        """
        :param algo_type: scg_md, scg mirror descent; cd, coordinate descent; scg_da, scg dual average
        """
        super(DualKSVM, self).__init__(lmda, gm, kernel, nsweep, b=b, c=c)
        # C = 1.0 / n  # box constraint on the dual
        self.obj_primal = []
        self.rho = rho
        self.algo_type = algo_type

    def fit(self, xtr, ytr, xte=None, yte=None):
        self.set_train_kernel(xtr)
        self.xtr = xtr
        self.ytr = ytr
        self.has_kte = not(xte is None)
        if self.nsweep is None:
            self.nsweep = xtr.shape[0]

        if not self.has_kte:
            # if self.algo_type == 'scg_md':
            #     self._stoc_coor_cython()
            if self.algo_type == 'cd':
                self.alpha, self.err_tr, self.obj, self.nker_opers = \
                    scg_svm.cd_svm(ktr=self.ktr, ytr=self.ytr, lmda=self.lmda, nsweep=np.int(self.nsweep))
            elif self.algo_type == 'scg_da':
                self.alpha, self.err_tr, self.obj, self.obj_primal, self.nker_opers, self.nnzs, = \
                    scg_svm.scg_da_svm(ktr=self.ktr, ytr=self.ytr, lmda=self.lmda,
                                       nsweep=np.int(self.nsweep), b=np.int(self.b), c=np.int(self.c))
        else:
            self.set_test_kernel(xtr, xte)
            self.yte = yte

            # if self.algo_type == 'scg_md':
            #     self._stoc_coor_cython()
            if self.algo_type == 'cd':
                self.alpha, self.err_tr, self.err_te, self.obj, self.nker_opers = \
                    scg_svm.cd_svm(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=yte, lmda=self.lmda, nsweep=np.int(self.nsweep))
            elif self.algo_type == 'scg_da':
                self.alpha, self.err_tr, self.err_te, self.obj, self.obj_primal, self.nker_opers, self.nnzs, = \
                    scg_svm.scg_da_svm(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=yte, lmda=self.lmda,
                                       nsweep=np.int(self.nsweep), b=np.int(self.b), c=np.int(self.c))
            # sys.exit(1)

    def train_test(self, xtr, ytr, xte, yte, algo_type="cy"):
        self.set_train_kernel(xtr)
        self.set_test_kernel(xtr, xte)
        self.has_kte = True
        self.ytr = ytr
        self.yte = yte
        if self.nsweep is None:
            self.nsweep = xtr.shape[0]
        if algo_type == "naive":
            self._rand_stoc_coor()
        elif algo_type == 'cy':
            # print " type "+str(self.nsweep.dtype)
            self._stoc_coor_cython()
        else:
            print "error"

    def predict(self, xte):
        self.set_test_kernel(self.xtr, xte)
        if self.alpha is None:
            print('need to train svm first')
            exit(1)
        pred = np.sign(self.kte.dot(self.alpha * self.ytr)).astype(int)
        return pred

    def _stoc_coor_cython(self):
        """
        call the cython wrapper
        """
        if not self.has_kte:
            self.alpha, self.err_tr, self.err_te, self.obj, self.obj_primal, self.nker_opers, self.nnzs, self.snorm_grad = \
                scg_svm.scg_md_svm(ktr=self.ktr, ytr=self.ytr, lmda=self.lmda,
                                  nsweep=np.int(self.nsweep), c=np.int(self.batchsize))
        else:
            self.alpha, self.err_tr, self.err_te, self.obj, self.obj_primal, self.nker_opers, self.nnzs, self.snorm_grad = \
                scg_svm.scg_md_svm(ktr=self.ktr, ytr=self.ytr, kte=self.kte, yte=self.yte, lmda=self.lmda,
                                nsweep=np.int(self.nsweep), c=np.int(self.batchsize))

    def _rand_stoc_coor(self):
        """
        stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
        variables
        """
        n = self.ktr.shape[0]
        T = n * self.nsweep - 1
        C = 1.0 / n
        yktr = (self.ytr[:, np.newaxis] * self.ktr) * self.ytr[np.newaxis, :]
        print"------------estimate parameters and set up variables----------------- "
        # comment:  seems very sensitive to the parameter estimation, if I use the second D_t, the algorithm diverges
        #
        lip = np.diag(yktr) / self.lmda
        l_max = np.max(lip)
        Q = 1
        D_t = Q * np.sqrt(1.0 / (2 * n))
        # D_t = Q * (n / 2) * C
        sig_list = self._esti_std(yktr)
        sig = np.sqrt((sig_list ** 2).sum())
        eta = np.ones(T + 1)
        eta *= np.minimum(1.0 / (2 * l_max), D_t / sig * np.sqrt(float(n) / (1 + T)))
        theta = eta + .0
        alpha = np.zeros(n)  # the most recent solution
        a_tilde = np.zeros(n)  # the accumulated solution in the path
        delta = np.zeros(T + 2)
        uu = np.zeros(n, dtype=int)
        # index of update, u[i] = t means the most recent update of
        # ith coordinate is in the t-th round, t = 0,1,...,T
        showtimes = 5
        t = 0
        count = 0
        print "estimated sigma: " + str(sig) + " lipschitz: " + str(l_max)
        print "----------------------start the algorithm----------------------"
        for i in range(self.nsweep):
            # index of batch data to compute stochastic coordinate gradient
            samp = np.random.choice(n, size=(n, self.batchsize))
            # samp = np.random.permutation(n)
            # index of sampled coordinate to update

            perm = np.random.permutation(n)
            for j in range(n):
                # samp_ind = samp[j, :]
                samp_ind = np.take(samp, j, axis=0)
                # samp_ind = samp[j]
                var_ind = perm[j]
                # var_ind = samp_ind
                delta[t + 1] = delta[t] + theta[t]
                subk = yktr[var_ind, samp_ind]
                # stoc_coor_grad = np.dot(subk, alpha[samp_ind]) * float(n) / self.batchsize - 1
                stoc_coor_grad = 1 / self.lmda * (np.dot(subk, alpha.take(samp_ind)) * float(n) / self.batchsize) - 1
                a_tilde[var_ind] += (delta[t + 1] - delta.take(uu.take(var_ind))) * alpha.take(var_ind)
                res = alpha.take(var_ind) - eta[t] * stoc_coor_grad
                if res < 0:
                    alpha[var_ind] = 0
                elif res <= C:
                    alpha[var_ind] = res
                else:
                    alpha[var_ind] = C
                # alpha[var_ind] = np.minimum(np.maximum(0, alpha[var_ind] - eta[t]*stoc_coor_grad), C)
                # alpha[var_ind] = self._prox_mapping(g=stoc_coor_grad, x0=alpha[var_ind], r=eta[t], C)
                # assert(all(0 <= x <= C for x in np.nditer(alpha[var_ind])))  #only works for size 1
                uu[var_ind] = t + 1
                t += 1
                count += self.batchsize
            if i % (self.nsweep / showtimes) == 0:
                print "# of sweeps " + str(i)
            # -------------compute the result after the ith sweep----------------
            if i % n == 0:
                a_avg = a_tilde + (delta[t] - delta.take(uu)) * alpha
                a_avg /= delta[t]
                # a_avg = alpha
                # assert(all(0 <= x <= C for x in np.nditer(a_avg)))
                yka = np.dot(yktr, a_avg)
                res = 1.0 / self.lmda * 0.5 * np.dot(a_avg, yka) - a_avg.sum()
                self.obj.append(res)
                # if i > 2 and self.obj[-1] > self.obj[-2]:
                # print "warning"
                nnzs = (a_avg != 0).sum()
                self.nnzs.append(nnzs)
                err = np.mean(yka < 0)
                self.err_tr.append(err)
                self.ker_oper.append(count)
                if self.has_kte:
                    pred = np.sign(np.dot(self.kte, self.ytr * a_avg))
                    err = np.mean(self.yte != pred)
                    self.err_te.append(err)
        # -------------compute the final result after nsweep-th sweeps---------------
        a_tilde += (delta[T + 1] - delta[uu]) * alpha
        self.alpha = a_tilde / delta[T + 1]
        self.final = self.lmda * (0.5 * np.dot(self.alpha, np.dot(yktr, self.alpha)) - self.alpha.sum())
        self.bound1 = (n - 1) * 0.5 * l_max / self.lmda
        self.bound2 = l_max
        self.bound3 = sig * np.sqrt(2)


    def _prox_mapping(self, g, x0, r, C):
        """
        proximal coordinate gradient mapping
        argmin  x*g + 1/r*D(x0,x)
        """
        x = x0 - r * g
        x = np.minimum(np.maximum(0, x), C)
        return x

    def _esti_std(self, kk):
        """
        estimate standard deviation of coordiante stochastic gradient
        """
        n = kk.shape[0]
        sig = np.zeros(n)
        alpha = np.random.uniform(0, 1.0/n, n)
        rep = 100
        for i in range(n):
            g = kk[i, :] / self.lmda * 1.0/n
            sig[i] = np.std(g) * n / np.sqrt(self.batchsize)
        return sig

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
