__author__ = 'qdpercy'

from mysvm import *
import time
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
        rec_step = 1
        t = 0
        flag = np.zeros(n)
        num_sv = 0
        count = 0
        num_hit = 0
        start = time.time()
        for k in range(self.nsweep):
            perm = np.random.permutation(n)
            for j in range(n):
                t += 1
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
                if t == rec_step:
                    rec_step *= 2
                    self.nker_opers.append(count)
                    yka = np.dot(yktr, alpha/(self.lmda * t))
                    self.err_tr.append(np.mean(yka < 0))
                    obj = 1.0/n * np.maximum(1-yka, 0).sum() + self.lmda/2*np.dot(alpha/(self.lmda * t), yka)
                    self.obj.append(obj)
                    self.nnzs.append(num_sv)
                    if self.has_kte:
                        pred = np.sign(np.dot(self.kte, self.ytr*alpha/(self.lmda * t)))
                        self.err_te.append(np.mean(self.yte != pred))
            # if k % (self.nsweep) == 0:
            print "# of sweeps " + str(k)
        print "num of hit %d" % num_hit
        print "time cost %f " % (time.time()-start)
        self.alpha = alpha / (self.lmda * t)


if __name__ == "__main__":
    if os.name == "nt":
        dtpath = "..\\..\\dataset\\ucibenchmark\\"
    elif os.name == "posix":
        dtpath = '../../dataset/benchmark_uci/'
    filename = 'splicemat.mat'
    filename = 'bananamat'
    data = scipy.io.loadmat(dtpath + filename)
