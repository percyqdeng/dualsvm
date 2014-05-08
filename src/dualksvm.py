__author__ = 'qdengpercy'

import scipy.io
import numpy as np
from sklearn import svm


class DualCoorDescent:
    """Dual randomized stochastic coordinate descent for kenel method, SVM and ridge regression

    lmda: int, regularizer
    alpha:  array type, weights of kernel classifier
    max_iter : int, maximal number of iteration
    C : anoter regularizer, essentially C=1/lmda*n
    kernel : string type,
        "rbf" : gaussian kernel
        "poly" : polynomial kernel
    """

    def __init__(self, n, lmda, gm=1, kernel='rbf', max_iter=1000, batchsize=2):
        self.batchsize = batchsize
        self.alpha = np.zeros(n)
        self.lmda = lmda
        self.c = 1.0 / (n * lmda)
        self.dim = n
        self.gamma = gm
        self.max_iter = max_iter
        self.kernel = kernel
        self.obj = []
        # pass

    def train(self, x, y, kernel='rbf'):
        K = self._set_kernels(x)
        # self.alpha = np.zeros(n)
        pass

    def test(self, x, y):

        pass

    def rand_stoc_coor(self, y, K):
        """
        average mirror descent for kernel svm
        """
        n = K.shape[0]
        K = (y[:, np.newaxis] * K) * y[np.newaxis, :]
        lip = np.sum(K, 1) + 1
        L_t = np.max(lip)
        Q = 1
        D_t = Q * self.dim * self.c ** 2 / 2
        sig = self._esti_para(K)
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
            samp_index = np.random.choice(n, size=self.batchsize, replace=False)
            var_ind = np.random.randint(n, size=1)
            theta_sum[t + 1] = theta_sum[t] + theta
            alpha_sum[samp_index] = (theta_sum[t] - theta_sum[u[samp_index] - 1]) * alpha[samp_index]
            subK = K[samp_index, samp_index]
            stoc_coor_grad = np.dot(subK, alpha[samp_index]) - alpha[samp_index]
            alpha[samp_index] = self._prox_mapping(stoc_coor_grad, alpha[samp_index], gamma)
            u[samp_index] = t + 1
            Z += theta
            if t % n == 0:
                tmp = alpha_sum + (theta_sum[t + 1] - theta_sum[u - 1]) * alpha
                alpha_avg = tmp / Z
                res = 0.5 * np.dot(np.dot(alpha_avg, K), alpha_avg) - alpha_avg.sum()
                self.obj.append(res)

        # averaging
        alpha_sum += (theta_sum[self.max_iter] - theta_sum[u - 1]) * alpha
        self.alpha = alpha_sum / Z

    def _set_kernels(self, x):
        if self.kernel == 'rbf':
            std = np.std(x, axis=0)
            x = x / std[np.newaxis, :]
            xsquare = np.sum(x**2, 1)
            xxT = np.dot(x, x.T)
            dist = xsquare[:, np.newaxis] -2 * xxT + xsquare[np.newaxis, :]
            K = np.exp(-self.gm * dist)
        else:
            print "the other kernel tbd"
        return K

    def _prox_mapping(self, v, x0, gamma):
        """
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

def test_dualsvm(data):
    x = data['x']
    y = data['t']
    y = np.ravel(y)
    trInd = data['train'] - 1
    teInd = data['test'] - 1
    rep = trInd.shape[0]
    rep = 1
    libsvm_err = np.zeros(rep)
    for i in range(rep):


def test_benchmark(data):
    x = data['x']
    y = data['t']
    y = np.ravel(y)
    trInd = data['train'] - 1
    teInd = data['test'] - 1
    rep = trInd.shape[0]
    rep = 10
    libsvm_err = np.zeros(rep)
    for i in range(rep):
        if i % 10 == 0:
            print "iteration #: "+str(i)


        clf = svm.SVC(kernel='rbf')
        clf.fit(x[trInd[i, :], :], y[trInd[i, :]])
        pred = clf.predict(x[teInd[i, :], :])
        # pred = np.ravel(pred)
        libsvm_err[i] = np.mean(pred != y[teInd[i, :]])

    return libsvm_err

if __name__ == "__main__":
    datapath = "/Users/qdengpercy/workspace/dataset/benchmark_uci/"
    filename = 'bananamat.mat'
    data = scipy.io.loadmat(datapath + filename)
    err = test_benchmark(data)
