__author__ = 'qdengpercy'

import scipy.io
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import numpy.linalg as la


class DualKSVM:
    """Dual randomized stochastic coordinate descent for kenel method, SVM and ridge regression

<<<<<<< HEAD
    lmda: int, regularizer
    alpha:  array type, weights of kernel classifier
    max_iter : int, maximal number of iteration
    C : anoter regularizer, essentially C=1/lmda*n

    kernel : string type,
        "rbf" : gaussian kernel
        "poly" : polynomial kernel
    gamma : parameter in rbf kernel
    """

    def __init__(self, n, lmda=0.01, gm=1, kernel='rbf', max_iter=1000, batchsize=2):
=======
	lmda: int, regularizer
	alpha:  array type, weights of kernel classifier
	max_iter : int, maximal number of iteration
	C : anoter regularizer, essentially C=1/lmda*n

	kernel : string type,
		"rbf" : gaussian kernel
		"poly" : polynomial kernel
	gm : parameter in rbf kernel

	err_tr
	"""

    def __init__(self, n, lmda=0.01, gm=1, kernel='rbf', nsweep=1000, batchsize=2):
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7
        self.batchsize = batchsize
        self.alpha = np.zeros(n)
        self.lmda = lmda
        self.c = 1.0 / (n * lmda)
        self.dim = n
<<<<<<< HEAD
        self.gamma = gm
        self.max_iter = max_iter
        self.kernel = kernel
        self.obj = []
        self.nnz = []
=======
        self.gm = gm
        self.nsweep = nsweep
        self.max_iter = nsweep * n
        self.kernel = kernel
        self.obj = []
        self.nnz = []
        self.err_tr = []
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7

    def train(self, x, y):
        K = self._set_kernels(x)
        self.rand_stoc_coor(y, K)

    def test(self, x, y):

        pass

    def rand_stoc_coor(self, y, K):
        """
<<<<<<< HEAD
        stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
        variables
        """
        n = K.shape[0]
        yKy = (y[:, np.newaxis] * K) * y[np.newaxis, :]
        lip = np.sum(K, 1) + 1
        L_t = np.max(lip)
        Q = 1
        D_t = Q * self.dim * self.c ** 2 / 2
        sig = self._esti_para(yKy)
        gamma = np.minimum(1 / (2 * L_t), D_t / sig * np.sqrt(self.dim / self.max_iter))
        theta = gamma
        alpha = np.zeros(n)
        Z = 0
        theta_sum = np.zeros(self.max_iter + 1)  # accumulated weight
        alpha_sum = np.zeros(n)
        u = np.ones(n)
        """index of the active iteration for each coordinate, u[i] = t means the most recent update of
        ith coordinate is in the t round
        """
        for t in range(0, self.max_iter):
            samp_ind = np.random.choice(n, size=self.batchsize, replace=False) # index of samples to compute stochastic coordinate gradient
            var_ind = np.random.randint(n, size=1)  # index of sampled coordinate to update
            theta_sum[t + 1] = theta_sum[t] + theta
            alpha_sum[var_ind] = (theta_sum[t] - theta_sum[u[var_ind] - 1]) * alpha[var_ind]
            subK = yKy[var_ind, samp_ind]
            stoc_coor_grad = np.dot(subK, alpha[samp_ind]) - alpha[samp_ind]
            alpha[var_ind] = self._prox_mapping(stoc_coor_grad, alpha[var_ind], gamma)
            u[var_ind] = t + 1
            Z += theta
            if t % n == 0:
                #compute dual objective
                tmp = alpha_sum + (theta_sum[t + 1] - theta_sum[u - 1]) * alpha
                alpha_avg = tmp / Z
                res = self.lmda * (0.5 * np.dot(np.dot(alpha_avg, yKy), alpha_avg) - alpha_avg.sum())
                self.obj.append(res)
                nnzs = (alpha_avg != 0).sum()
                self.nnz.append(nnzs)

        # averaging
        alpha_sum += (theta_sum[self.max_iter] - theta_sum[u - 1]) * alpha
        self.alpha = alpha_sum / Z

    def plot_train_result(self):
        row = 1
=======
		stochastic coordinate descent on the dual svm, random sample a batch of data and update on another random sampled
		variables
		"""
        n = K.shape[0]
        yKy = (y[:, np.newaxis] * K) * y[np.newaxis, :]
        lip = np.diag(yKy)
        L_t = np.max(lip)
        Q = 1
        D_t = Q * self.dim * self.c ** 2 / 2
        sig = self._esti_std(yKy)
        gamma = np.minimum(1.0 / (2 * L_t), D_t / sig * np.sqrt(float(self.dim) / self.max_iter))
        theta = gamma
        alpha = np.zeros(n)
        Z = 0
        theta_sum = np.zeros(self.max_iter + 2)  # accumulated weight
        alpha_sum = np.zeros(n)
        uu = np.ones(n, dtype=int)
        # index of the active iteration for each coordinate, u[i] = t means the most recent update of
        # ith coordinate is in the t round
        print "gamma: " + str(gamma) + " theta: " + str(theta)
        t = 0
        for i in range(self.nsweep):
            samp = np.random.choice(n, size=(n, self.batchsize))  # index of samples to compute stochastic coordinate gradient
            samp = range(n)
            perm = np.random.permutation(n)  # index of sampled coordinate to update
            for j in range(n):
                # samp_ind = samp[j, :]
                samp_ind = samp
                var_ind = perm[j]
                theta_sum[t + 1] = theta_sum[t] + theta
                alpha_sum[var_ind] += (theta_sum[t] - theta_sum[(uu[var_ind] - 1)]) * alpha[var_ind]
                subK = yKy[var_ind, samp_ind]
                stoc_coor_grad = np.dot(subK, alpha[samp_ind])*float(n)/self.batchsize - 1
                alpha[var_ind] = self._prox_mapping(stoc_coor_grad, alpha[var_ind], gamma)
                uu[var_ind] = t + 1
                Z += theta
                t += 1
            print "# of sweeps " + str(i)
            tmp = alpha_sum + (theta_sum[t + 1] - theta_sum[(uu - 1)]) * alpha
            alpha_avg = tmp / Z
            yHa = np.dot(yKy, alpha_avg)
            res = self.lmda * (0.5 * np.dot(alpha_avg, yHa) - alpha_avg.sum())
            self.obj.append(res)
            nnzs = (alpha_avg != 0).sum()
            self.nnz.append(nnzs)
            err = np.mean(yHa > 0)
            self.err_tr.append(err)


        # averaging
        alpha_sum += (theta_sum[self.max_iter] - theta_sum[uu - 1]) * alpha
        self.alpha = alpha_sum / Z

    def plot_train_result(self):
        row = 2
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7
        col = 2
        plt.subplot(row, col, 1)
        plt.plot(self.obj, 'bx-', label="objective")
        plt.subplot(row, col, 2)
        plt.plot(self.nnz, 'bx-', label="# of nnzs")
<<<<<<< HEAD
=======
        plt.subplot(row, col, 3)
        plt.plot(self.err_tr, label="training error")
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7

    def _set_kernels(self, x):
        if self.kernel == 'rbf':
            std = np.std(x, axis=0)
            x = x / std[np.newaxis, :]
<<<<<<< HEAD
            xsquare = np.sum(x**2, 1)
=======
            xsquare = np.sum(x ** 2, 1)
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7
            xxT = np.dot(x, x.T)
            dist = xsquare[:, np.newaxis] - 2 * xxT + xsquare[np.newaxis, :]
            K = np.exp(-self.gm * dist)
        else:
            print "the other kernel tbd"
        return K

    def _prox_mapping(self, v, x0, gamma):
        """
<<<<<<< HEAD
        proximal coordinate gradient mapping
         argmin  x*v + 1/gamma*D(x0,x)
        """
        x = x0 - gamma * v
        x = self._proj_box(x)
        return x

    def _proj_box(self, x):
        """
        project x into [0,C] box
        """
        return np.minimum(np.maximum(0, x), self.c)

    def _esti_para(self, K):
        n = K.shape[0]
        pass
=======
		proximal coordinate gradient mapping
		argmin  x*v + 1/gamma*D(x0,x)
		"""
        x = x0 - gamma * v
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

        return np.sqrt((sig ** 2).sum())

>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7

def test_dualsvm(data):
    x = data['x']
    y = data['t']
    y = np.ravel(y)
    trInd = data['train'] - 1
    teInd = data['test'] - 1
<<<<<<< HEAD
    i = 1
    ntr = len(y[trInd[i, :]])
    xtr = x[trInd[i, :], :]
    ytr = y[trInd[i, :]]
    dsvm = DualKSVM(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', max_iter=10 * ntr, batchsize=5)
    dsvm.train(xtr, ytr)
    dsvm.plot_train_result()
=======
    i = np.random.choice(trInd.shape[0])
    ntr = len(y[trInd[i, :]])
    xtr = x[trInd[i, :], :]
    ytr = y[trInd[i, :]]
    dsvm = DualKSVM(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', nsweep=100, batchsize=5)
    dsvm.train(xtr, ytr)
    dsvm.plot_train_result()
    return dsvm
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7


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
<<<<<<< HEAD
        ytr =  y[trInd[i, :]]
        dsvm = DualKSVM(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', max_iter=10 * ntr, batchsize=5)
=======
        ytr = y[trInd[i, :]]
        dsvm = DualKSVM(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', nsweep=20, batchsize=5)
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7
        dsvm.train(xtr, ytr)
        clf = svm.SVC(kernel='rbf')
        clf.fit(xtr, ytr)
        pred = clf.predict(x[teInd[i, :], :])
        # pred = np.ravel(pred)
        libsvm_err[i] = np.mean(pred != y[teInd[i, :]])

    dsvm.plot_train_result()
    return libsvm_err
<<<<<<< HEAD
=======

>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7

if __name__ == "__main__":
    datapath = "/Users/qdengpercy/workspace/dataset/benchmark_uci/"
    filename = 'bananamat.mat'
    data = scipy.io.loadmat(datapath + filename)
<<<<<<< HEAD
    err = test_benchmark(data)
=======
    # err = test_benchmark(data)
    dsvm = test_dualsvm(data)
>>>>>>> 6cbaa3d8ffe1c603d21fa41cf6599958ed586dc7
