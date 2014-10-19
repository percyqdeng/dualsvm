
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import svm
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


# obtain uci dataset
dtname = ucifile[1] + '.mat'
dtpath = ucipath
data = scipy.io.loadmat(dtpath + dtname)
x = data['x']
        # self.x = preprocess_data(self.x)
# x = preprocessing.normalize(x, norm='l2')
y = data['t']
y = (np.squeeze(y)).astype(np.intc)
# train_ind = data['train'] - 1
# test_ind = data['test'] - 1

#obtain usps
# pos_class = 3
# neg_class = None
# data = load_usps()
# if neg_class is None:
#     x, y = convert_one_vs_all(data, pos_class)
#     one_vs_rest = True
# else:
#     x, y = convert_binary(data, pos_class, neg_class)
#     one_vs_rest = False
x = preprocessing.scale(x)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

C_list = np.array([1e-2, 1e-1, 1, 1e1, 1e2])
# C_list = np.array([1e-2, 1e-1])
gamma= 1
n_iter = 1
trainerr_dasvm = {}
testerr_dasvm = {}
trainerr_cd = {}
testerr_cd = {}
obj_dasvm = {}
obj_cd = {}
trainerr_pega = {}
testerr_pega = {}
obj_pega = {}

rs = cv.ShuffleSplit(x.shape[0], n_iter=n_iter, train_size=0.5, test_size=0.5, random_state=11)
for k, (train_index, test_index) in enumerate(rs):
    for j, C in enumerate(C_list):
        n = train_index.size
        clf = DualKSVM(lmda=C/n, kernel='rbf', gm=gamma, nsweep=int(1.5*n), rho=0.1, algo_type='scg_da')
        clf.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
        trainerr_dasvm[C] = clf.err_tr
        testerr_dasvm[C] = clf.err_te
        obj_dasvm[C] = clf.obj


        clf2 = DualKSVM(lmda=C/n, kernel='rbf', gm=gamma, nsweep=500, algo_type='cd')
        clf2.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
        trainerr_cd[C] = clf2.err_tr
        testerr_cd[C] = clf2.err_te
        obj_cd[C] = clf2.obj


        clf3 = Pegasos(lmda=C/n, gm=gamma, kernel='rbf', nsweep=5, batchsize=1)
        clf3.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
        trainerr_pega[C] = clf3.err_tr
        testerr_pega[C] = clf3.err_te
        obj_pega[C] = clf3.obj

row = 2
col = 2
plt.figure()
plt.subplot(row,col,1)
for i, c in trainerr_dasvm.iteritems():
    plt.semilogx(clf.nker_opers, trainerr_dasvm[i], 'x-')
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

# plt.xscale('log')

# plt.figure()
plt.subplot(row,col,2)
for i, c in testerr_dasvm.iteritems():
    plt.plot(clf.nker_opers, testerr_dasvm[i], 'x-')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# plt.xscale('log')

plt.subplot(row,col,3)
for i, c in obj_dasvm.iteritems():
    plt.plot(clf.nker_opers, obj_dasvm[i], 'x-', label='c=%3.3f' % i)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.legend(loc='best')


plt.figure()
plt.subplot(row,col,1)
for i, c in trainerr_cd.iteritems():
    plt.plot(clf2.nker_opers, trainerr_cd[i], 'x-')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.subplot(row,col,2)
for i, c in testerr_cd.iteritems():
    plt.plot(clf2.nker_opers, testerr_cd[i], 'x-')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.subplot(row,col,3)
for i, c in obj_cd.iteritems():
    plt.plot(clf2.nker_opers, obj_cd[i], 'o-')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))


plt.figure()
plt.subplot(row, col, 1)
for i, c in trainerr_pega.iteritems():
    plt.plot(clf3.nker_opers, trainerr_pega[i], 'x-')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.subplot(row,col,2)
for i, c in testerr_pega.iteritems():
    plt.plot(clf3.nker_opers, testerr_pega[i], 'x-')
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.subplot(row,col,3)
for i, c in obj_pega.iteritems():
    plt.plot(clf3.nker_opers, obj_pega[i], 'x-')

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
