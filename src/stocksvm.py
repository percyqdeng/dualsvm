__author__ = 'qdpercy'

from mysvm import *

class Pegasos(MySVM):

    def __init__(self, n, lmda=0.01, gm=1, kernel='rbf', nsweep=1000, batchsize=2):
        super(Pegasos, self).__init__(n, lmda, gm, kernel, nsweep, batchsize)

    def train(self, xtr, ytr):
        self.set_train_kernel(xtr)
        self.ytr = ytr
        self._kernel_primal_stoch()

    def train_test(self, xtr, ytr, xte, yte):
        self.set_train_kernel(xtr)
        self.set_test_kernel(xtr, xte)
        self.has_kte = True
        self.ytr = ytr
        self.yte = yte
        self._kernel_primal_stoch()

    def _kernel_primal_stoch(self):
        n = self.ktr.shape[0]
        yktr = (self.ytr[:, np.newaxis] * self.ktr) * self.ytr[np.newaxis, :]
        alpha = np.zeros(n)
        showtimes = 5
        t = 0
        flag = np.zeros(n)
        num_sv = 0
        count = 0
        for k in range(self.nsweep):
            perm = np.random.permutation(n)
            for j in range(n):
                t += 1
                i = perm[j]
                res = np.dot(yktr[i, :], alpha)/(self.lmda*t)
                count += num_sv
                if res < 1:
                    alpha[i] += 1
                    if flag[i] == 0:
                        flag[i] = 1
                        num_sv += 1
            if k % (self.nsweep / showtimes) == 0:
                print "# of sweeps " + str(k)
            # alpha /= self.lmda * self.T
            self.ker_oper.append(count)
            yka = np.dot(yktr, alpha/(self.lmda * t))
            self.err_tr.append(np.mean(yka < 0))
            obj = 1.0/n * np.maximum(1-yka, 0).sum() + self.lmda/2*np.dot(alpha/(self.lmda * t), yka)
            self.obj.append(obj)
            if self.has_kte:
                pred = np.sign(np.dot(self.kte, self.ytr*alpha/(self.lmda * t)))
                self.err_te.append(np.mean(self.yte != pred))
        self.alpha = alpha

    def plot_train_result(self):
        row = 1
        col = 2
        plt.figure()
        # plt.subplot(row, col, 1)
        plt.plot(self.obj, 'b-', label="pegasos")
        seq = range(self.dim, self.T+2, self.dim)
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


def test_pegasos(data):
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
    kpega = Pegasos(n=ntr, lmda=1.0 / ntr, gm=1, kernel='rbf', nsweep=20, batchsize=1)
    kpega.train_test(xtr, ytr, xte, yte)
    kpega.plot_train_result()
    return kpega


if __name__ == "__main__":
    if os.name == "nt":
        dtpath = "..\\..\\dataset\\ucibenchmark\\"
    elif os.name == "posix":
        dtpath = '../../dataset/benchmark_uci/'
    filename = 'splicemat.mat'
    filename = 'bananamat'
    data = scipy.io.loadmat(dtpath + filename)
    # err = test_benchmark(data)
    kpega = test_pegasos(data)