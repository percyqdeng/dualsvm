import sklearn.preprocessing as preprocessing
import sklearn.cross_validation as cv
import matplotlib.pyplot as plt
import time
from auxiliary.load_data import *
from dsvm.dualksvm import DualKSVM
from dsvm.stocksvm import Pegasos
"""
compare time cost of cd, pegasos
"""

if os.name == "nt":
    ucipath = "..\\dataset\\ucibenchmark\\"
    uspspath = "..\\dataset\\usps\\"
    mnistpath = "..\\dataset\\mnist\\"
elif os.name == "posix":
    ucipath = '../dataset/benchmark_uci/'
    uspspath = '../dataset/usps/'
    mnistpath = '../dataset/mnist/'
ucifile = ["bananamat", "breast_cancermat", "diabetismat", "flare_solarmat", "germanmat",
                "heartmat", "ringnormmat", "splicemat"]


# obtain uci dataset
# dtname = ucifile[1] + '.mat'
# dtpath = ucipath
# data = scipy.io.loadmat(dtpath + dtname)
# x = data['x']
#         # self.x = preprocess_data(self.x)
# # x = preprocessing.normalize(x, norm='l2')
# y = data['t']
# y = (np.squeeze(y)).astype(np.intc)
# # train_ind = data['train'] - 1
# # test_ind = data['test'] - 1

#obtain usps
pos_class = 5
neg_class = 6
data = load_usps()
if neg_class is None:
    x, y = convert_one_vs_all(data, pos_class)
    one_vs_rest = True
else:
    x, y = convert_binary(data, pos_class, neg_class)
    one_vs_rest = False
x = preprocessing.scale(x)
y = np.asarray(y, dtype=float)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
xsq = (x**2).sum(axis=1)
dist = xsq[:, np.newaxis] - 2*x.dot(x.T) + xsq[np.newaxis, :]
gamma = 3.0/np.median(dist)
# gamma = 0.4
print "x.shape: (%d, %d)" % (x.shape[0], x.shape[1])

C_list = np.array([1e-2, 1e-1, 1, 1e1, 1e2])
C = float(1)
# C_list = np.array([1e-2, 1e-1])
rho1 = 0.001
rho2 = 0.8
n_iter = 10
trainerr_dasvm = {}
testerr_dasvm = {}
obj_dasvm = {}
trainerr_dasvm2 = {}
testerr_dasvm2 = {}
obj_dasvm2 = {}
trainerr_cd = {}
testerr_cd = {}
obj_cd = {}
trainerr_pega = {}
testerr_pega = {}
obj_pega = {}

b = 4
c = 1
rs = cv.ShuffleSplit(x.shape[0], n_iter=n_iter, train_size=0.3, test_size=0.5, random_state=None)
for k, (train_index, test_index) in enumerate(rs):
    print "iter # %d" % k
    n = train_index.size
    start_t = time.time()
    clf = DualKSVM(lmda=C/n, kernelstr='rbf', gm=gamma, nsweep=n, b=b, c=c, verbose=True, rho=rho1, algo_type='scg_da')
    clf.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
    if C in trainerr_dasvm:
        trainerr_dasvm[C] = np.vstack((trainerr_dasvm[C], clf.err_tr))
        testerr_dasvm[C] = np.vstack((testerr_dasvm[C], clf.err_te))
        obj_dasvm[C] = np.vstack((obj_dasvm[C], clf.obj))
    else:
        trainerr_dasvm[C] = clf.err_tr
        testerr_dasvm[C] = clf.err_te
        obj_dasvm[C] = clf.obj
    print "dualsvm time %f " % (time.time()-start_t)

    start_t = time.time()
    clf2 = DualKSVM(lmda=C/n, kernelstr='rbf', gm=gamma, nsweep=int(4*n), algo_type='cd')
    clf2.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
    if C in trainerr_cd:
        trainerr_cd[C] = np.vstack((trainerr_cd[C], clf2.err_tr))
        testerr_cd[C] = np.vstack((testerr_cd[C], clf2.err_te))
        obj_cd[C] = np.vstack((obj_cd[C], clf2.obj))
    else:
        trainerr_cd[C] = clf2.err_tr
        testerr_cd[C] = clf2.err_te
        obj_cd[C] = clf2.obj
    print "cd svm time %f" % (time.time() - start_t)

    start_t = time.time()
    clf3 = Pegasos(lmda=C/n, gm=gamma, kernelstr='rbf', nsweep=5, batchsize=1)
    clf3.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
    if C in trainerr_pega:
        trainerr_pega[C] = np.vstack((trainerr_pega[C], clf3.err_tr))
        testerr_pega[C] = np.vstack((testerr_pega[C], clf3.err_te))
        obj_pega[C] = np.vstack((obj_pega[C], clf3.obj))
    else:
        trainerr_pega[C] = clf3.err_tr
        testerr_pega[C] = clf3.err_te
        obj_pega[C] = clf3.obj
    print "pegasos time %f" % (time.time() - start_t)

    # start_t = time.time()
    # clf4 = DualKSVM(lmda=C/n, kernel='rbf', gm=gamma, nsweep=n, rho=rho2, algo_type='scg_da')
    # clf4.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
    # if C in trainerr_dasvm2:
    #     trainerr_dasvm2[C] = np.vstack((trainerr_dasvm2[C], clf4.err_tr))
    #     testerr_dasvm2[C] = np.vstack((testerr_dasvm2[C], clf4.err_te))
    #     obj_dasvm2[C] = np.vstack((obj_dasvm2[C], clf4.obj))
    # else:
    #     trainerr_dasvm2[C] = clf4.err_tr
    #     testerr_dasvm2[C] = clf4.err_te
    #     obj_dasvm2[C] = clf4.obj
    # print "da svm time %f" % (time.time() - start_t)

row = 2
col = 2
plt.figure()
plt.errorbar(clf.nker_opers, obj_dasvm[C].mean(axis=0), yerr=obj_dasvm[C].std(axis=0), fmt='--', label=r'scg-svm')
plt.errorbar(clf2.nker_opers, obj_cd[C].mean(axis=0), yerr=obj_cd[C].std(axis=0), fmt='.-', label='cd-svm')
# plt.errorbar(clf4.nker_opers, obj_dasvm2[C].mean(axis=0), yerr=obj_dasvm2[C].std(axis=0), fmt='--', label=r'scg, $\rho=$%2.2f' % rho2)
plt.legend(loc='best')
plt.title(r'C=%2.4f, $\gamma=%2.2f$' % (C, gamma))
plt.xscale('log')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel('number of kernel evaluations')
plt.ylabel('dual objective')
plt.savefig('svmtest/output/ksvm_usps_conv_%d_%d_C_%2.2f_gm_%2.2f.eps' %(pos_class, neg_class, C, gamma))

# plt.xscale('log')

plt.figure()
# plt.subplot(row,col,2)
plt.errorbar(clf.nker_opers, testerr_dasvm[C].mean(axis=0), yerr=testerr_dasvm[C].std(axis=0), fmt='x-', label=r'scg-svm')
plt.errorbar(clf2.nker_opers, testerr_cd[C].mean(axis=0), yerr=testerr_cd[C].std(axis=0), fmt='o-', label='cd-svm')
plt.errorbar(clf3.nker_opers, testerr_pega[C].mean(axis=0), yerr=testerr_pega[C].std(axis=0), fmt='D-', label='Pegasos')
plt.ylabel('test error')
plt.xlabel('number of kernel evaluations')
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.savefig('svmtest/output/usps_train_error%d_%d_C_%2.2f_gm_%2.2f.eps' %(pos_class, neg_class, C, gamma))

plt.figure()
plt.errorbar(clf.nker_opers, trainerr_dasvm[C].mean(axis=0), yerr=trainerr_dasvm[C].std(axis=0), fmt='x-', label=r'scg-svm')
plt.errorbar(clf2.nker_opers, trainerr_cd[C].mean(axis=0), yerr=trainerr_cd[C].std(axis=0), fmt='o-', label='cd-svm')
plt.errorbar(clf3.nker_opers, trainerr_pega[C].mean(axis=0), yerr=trainerr_pega[C].std(axis=0), fmt='D-', label='Pegasos')
plt.ylabel('train error')
# plt.errorbar(clf4.nker_opers, testerr_dasvm2[C].mean(axis=0), yerr=testerr_dasvm2[C].std(axis=0), fmt='s-', label=r'scg, $\rho=$%2.2f' %rho2)
plt.xlabel('number of kernel evaluations')
plt.title(r'C=%2.4f, $\gamma=%2.2f$' % (C, gamma))
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1)
plt.savefig('svmtest/output/usps_test_error%d_%d_C_%2.2f_gm_%2.2f.eps' %(pos_class, neg_class, C, gamma))

# plt.errorbar(clf_da.nker_opers, testerr_dasvm[C].mean(), yerr=testerr_dasvm[C].std(axis=0), 'x-', label='scg')
# plt.semilogy(clf2.nker_opers, testerr_cd[C], 'o-', label='cd')
# plt.semilogy(clf_pega.nker_opers, testerr_pega[C], 'D-', label='Pegasos')
