
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import time
import numpy as np
from load_data import *
from dualksvm import *
from stocksvm import *


if os.name == "nt":
    ucipath = "..\\..\\dataset\\ucibenchmark\\"
    uspspath = "..\\..\\dataset\\usps\\"
    mnistpath = "..\\..\\dataset\\mnist\\"
elif os.name == "posix":
    ucipath = '../../dataset/benchmark_uci/'
    uspspath = '../../dataset/usps/'
    mnistpath = '../../dataset/mnist/'
ucifile = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]
dtname = ucifile[1] + '.mat'
dtpath = ucipath
data = scipy.io.loadmat(dtpath + dtname)
x = data['x']
        # self.x = preprocess_data(self.x)
y = data['t']
y = (np.squeeze(y)).astype(np.intc)
# train_ind = data['train'] - 1
# test_ind = data['test'] - 1

C_list = np.array([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])
gamma_list = np.array([1e-1, 1, 5])
gamma_list = [1]
n_iter = 4
test_err_libsvm = np.zeros((len(gamma_list), len(C_list), n_iter))
test_err_pegasos = np.zeros((len(gamma_list), len(C_list), n_iter))
test_err_da_svm = np.zeros((len(gamma_list), len(C_list), n_iter))
test_err_cd_svm = np.zeros((len(gamma_list), len(C_list), n_iter))
rs = cv.ShuffleSplit(x.shape[0], n_iter=n_iter, train_size=0.6, test_size=0.4, random_state=11)
for k, (train_index, test_index) in enumerate(rs):
    xtrain = x[train_index, :]
    ytrain = y[train_index]
    xtest = x[test_index, :]
    ytest = y[test_index]
    ntrain = ytrain.size
    n_folds = 4
    kf = cv.KFold(ntrain, n_folds=n_folds)
    for tr_ind, val_ind in kf:
        x1 = xtrain[tr_ind, :]
        y1 = ytrain[tr_ind]
        x2 = xtrain[val_ind, :]
        y2 = ytrain[val_ind]

    gamma, C = DualKSVM.
    for i, gamma in enumerate(gamma_list):
        for j, C in enumerate(C_list):
            clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            clf.fit(x[train_index,:], y[train_index])
            pred = clf.predict(x[test_index, :])
            test_err_libsvm[i, j, k] = zero_one_loss(y[test_index], pred)
            n = len(train_index)
            # clf = DualKSVM(lmda=C/n, kernel='rbf', gm=gamma, algo_type='scg_md')
            # clf.fit(x[train_index, :], y[train_index])
            # pred = clf.predict(x[test_index, :])
            # test_err_md_svm[i, j, k] = zero_one_loss(y[test_index], pred)

            clf = DualKSVM(lmda=C/n, kernel='rbf', gm=gamma, rho=1, algo_type='cd')
            clf.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
            pred = clf.predict(x[test_index, :])
            test_err_cd_svm[i, j, k] = zero_one_loss(y[test_index], pred)

            clf = DualKSVM(lmda=C/n, kernel='rbf', gm=gamma, algo_type='scg_da')
            clf.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
            pred = clf.predict(x[test_index, :])
            test_err_da_svm[i, j, k] = zero_one_loss(y[test_index], pred)

            clf4 = Pegasos(lmda=C/n, gm=gamma, kernel='rbf', nsweep=4,batchsize=1)
            clf4.fit(x[train_index, :], y[train_index], x[test_index,:], y[test_index])
            pred = clf.predict(x[test_index, :])
            test_err_pegasos[i, j, k] = zero_one_loss(y[test_index], pred)

err_libsvm = test_err_libsvm.mean(axis=2)
err_pegasos = test_err_pegasos.mean(axis=2)
err_da = test_err_da_svm.mean(axis=2)
err_cd = test_err_cd_svm.mean(axis=2)

plt.figure()
plt.plot(clf.err_tr)
plt.figure()
plt.plot(clf.obj, label='dual objective')
plt.plot(clf.obj_primal, label='primal objective')
plt.legend()