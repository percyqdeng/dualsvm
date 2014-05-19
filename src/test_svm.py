__author__ = 'qdengpercy'

from dualksvm import *
from stocksvm import *
from sklearn import svm
import os
class Test_SVM(object):
    """
    Test_SVM
    """
    def __init__(self, dtname='bananamat.mat'):
        if os.name == "nt":
            dtpath = "..\\..\\dataset\\ucibenchmark\\"
        elif os.name == "posix":
            dtpath = '../../dataset/benchmark_uci/'
        data = scipy.io.loadmat(dtpath + dtname)
        self.x = data['x']
        # self.x = preprocess_data(self.x)
        self.y = data['t']
        self.y = np.squeeze(self.y)
        self.train_ind = data['train'] - 1
        self.test_ind = data['test'] - 1
        print "use dataset: "+str(dtname)
        print "n = "+str(self.x.shape[0])+" p = "+str(self.x.shape[1])

    def rand_cmp_svm(self):
        rep = self.train_ind.shape[0]
        i = np.random.randint(low=0, high=rep)
        xtr = self.x[self.train_ind[i, :], :]
        ytr = self.y[self.train_ind[i, :]]
        xte = self.x[self.test_ind[i, :], :]
        yte = self.y[self.test_ind[i, :]]
        ntr = xtr.shape[0]
        lmd = 1.0
        # lmda=100 / float(ntr)
        dsvm = DualKSVM(n=ntr, lmda=lmd, gm=1, kernel='rbf', nsweep=20*ntr, batchsize=1)
        dsvm.train_test(xtr, ytr, xte, yte)
        kpega = Pegasos(n=ntr, lmda=lmd, gm=1, kernel='rbf', nsweep=20, batchsize=1)
        kpega.train_test(xtr, ytr, xte, yte)

        plt.subplot(2,2,1)
        plt.plot(dsvm.ker_oper, dsvm.err_tr, 'r-', label="dualcoor")
        plt.plot(kpega.ker_oper, kpega.err_tr, 'b-', label="pegasos")
        plt.legend()
        plt.ylabel("train err")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # plt.figure()
        plt.subplot(2, 2, 2)
        plt.plot(dsvm.ker_oper, dsvm.err_te, 'r-', label="dualcoor")
        plt.plot(kpega.ker_oper, kpega.err_te, 'b-', label="pegasos")
        plt.legend()
        plt.ylabel("test err")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()

        plt.figure()
        # plt.subplot(2,2,3)
        plt.plot(dsvm.ker_oper, np.log(-np.asarray(dsvm.obj)), 'r-', label="dualcoor")

        # plt.subplot(122)
        plt.plot(kpega.ker_oper, np.log(kpega.obj), 'b-', label="pegasos")
        plt.legend(loc='4')
        plt.ylabel("obj")
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.tight_layout()

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
    os.system("%load_ext autoreload")
    if os.name == "nt":
        dtpath = "..\\..\\dataset\\ucibenchmark\\"
    elif os.name == "posix":
        dtpath = '../../dataset/benchmark_uci/'
    filename = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
    # dsvm = test_dualsvm(data)
    newtest = Test_SVM(filename[2])
    newtest.rand_cmp_svm()
