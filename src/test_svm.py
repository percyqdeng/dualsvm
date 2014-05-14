__author__ = 'qdengpercy'

from dualksvm import *
import os
class Test_SVM(object):
    """
    Test_SVM
    """
    def __init__(self, dtname='bananamat.mat'):
        if os.name == "nt":
            dtpath = "..\\dataset\\"
        elif os.name == "posix":
            dtpath = '/Users/qdengpercy/workspace/dataset/benchmark_uci/'
        data = scipy.io.loadmat(dtpath + dtname)
        self.x = data['x']
        # self.x = preprocess_data(self.x)
        self.y = data['t']
        self.y = np.squeeze(self.y)
        self.train_ind = data['train'] - 1
        self.test_ind = data['test'] - 1
        print "use dataset: "+str(dtname)
        print "n = "+str(self.x.shape[0])+" p = "+str(self.x.shape[1])

    def rand_test_svm(self):
        rep = self.train_ind.shape[0]
        i = np.random.randint(low=0, high=rep)
        xtr = self.x[self.train_ind[i, :], :]
        ytr = self.y[self.train_ind[i, :]]
        xte = self.x[self.test_ind[i, :], :]
        yte = self.y[self.test_ind[i, :]]
        ntr = xtr.shape[0]
        self.dsvm = DualKSVM(n=ntr, lmda=.001)
