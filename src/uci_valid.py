
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import svm
from tune_hyper import *
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
dtname = ucifile[5] + '.mat'
dtpath = ucipath
data = scipy.io.loadmat(dtpath + dtname)
x = data['x']
        # self.x = preprocess_data(self.x)
y = data['t']
y = (np.squeeze(y)).astype(np.intc)
# train_ind = data['train'] - 1
# test_ind = data['test'] - 1
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

C_list = np.array([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])
gamma_list = np.array([1e-1, 1, 5])
# gamma_list = [1]
n_iter = 4
err_libsvm = np.zeros( n_iter)
err_pegasos = np.zeros(n_iter)
err_da_svm = np.zeros(n_iter)
err_cd_svm = np.zeros(n_iter)
err_boost = np.zeros(n_iter)
rs = cv.ShuffleSplit(x.shape[0], n_iter=n_iter, train_size=0.5, test_size=0.5, random_state=11)
for k, (train_index, test_index) in enumerate(rs):
    xtrain = x[train_index, :]
    ytrain = y[train_index]
    xtest = x[test_index, :]
    ytest = y[test_index]
    ntrain = ytrain.size

    gamma, C = DualKSVM.tune_parameter(xtrain, ytrain, gmlist=gamma_list, Clist=C_list, algo_type='scg_da')
    clf = DualKSVM(lmda=C/ntrain, gm=gamma, rho=0.1, verbose=False,algo_type='scg_da')
    clf.fit(xtrain, ytrain)
    pred = clf.predict(xtest)
    err_da_svm[k] = zero_one_loss(pred, ytest)
    print 'dasvm gamma:%f C:%f' % (gamma, C)

    gamma, C = DualKSVM.tune_parameter(xtrain, ytrain,gmlist=gamma_list, Clist=C_list, algo_type='cd')
    clf2 = DualKSVM(lmda=C/ntrain, gm=gamma, verbose=False, algo_type='cd')
    clf2.fit(xtrain, ytrain)
    pred = clf2.predict(xtest)
    err_cd_svm[k] = zero_one_loss(pred, ytest)
    print 'cdsvm gamma:%f C:%f' % (gamma, C)

    gamma, C = Pegasos.tune_parameter(xtrain, ytrain,gmlist=gamma_list, Clist=C_list)
    clf3 = Pegasos(lmda=C/ntrain, gm=gamma, verbose=False)
    clf3.fit(xtrain, ytrain)
    pred = clf3.predict(xtest)
    err_pegasos[k] = zero_one_loss(pred, ytest)
    print 'pegasos gamma:%f C:%f' % (gamma, C)

    gamma, C = tune_libsvm(xtrain, ytrain, gmlist=gamma_list, Clist=C_list)
    clf4 = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf4.fit(xtrain, ytrain)
    pred = clf4.predict(xtest)
    err_libsvm[k] = zero_one_loss(pred, ytest)
    print 'libsvm gamma:%f C:%f' % (gamma, C)

    T = tune_adaboost(xtrain, ytrain)
    clf5 = AdaBoostClassifier(n_estimators=T)
    clf5.fit(xtrain, ytrain)
    pred = clf5.predict(xtest)
    err_boost[k] = zero_one_loss(pred, ytest)

err = np.array([err_da_svm.mean(), err_cd_svm.mean(), err_pegasos.mean(), err_libsvm.mean(), err_boost.mean()])


# plt.figure()
# plt.plot(clf.err_tr)
# plt.figure()
# plt.plot(clf.obj, label='dual objective')
# plt.plot(clf.obj_primal, label='primal objective')
# plt.legend()