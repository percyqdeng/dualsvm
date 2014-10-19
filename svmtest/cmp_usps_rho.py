import sklearn.preprocessing as preprocessing
import sklearn.cross_validation as cv
import matplotlib.pyplot as plt
import time
from auxiliary.load_data import *
from dsvm.dualksvm import DualKSVM
from dsvm.stocksvm import Pegasos
# dreload()
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

# -----------------obtain uci dataset------------------
# dtname = ucifile[2] + '.mat'
# dtpath = ucipath
# data = scipy.io.loadmat(dtpath + dtname)
# x = data['x']
# y = np.asarray(np.squeeze(data['t']), dtype=float)
# # self.x = preprocess_data(self.x)
# # x = preprocessing.normalize(x, norm='l2')
# y = (np.squeeze(y)).astype(np.intc)
# # train_ind = data['train'] - 1
# # test_ind = data['test'] - 1

#-----------obtain usps---------------
pos_class = 5
neg_class = 6
data = load_usps()
if neg_class is None:
    x, y = convert_one_vs_all(data, pos_class)
    one_vs_rest = True
else:
    x, y = convert_binary(data, pos_class, neg_class)
    one_vs_rest = False

# --------------------preprocess------------------
x = preprocessing.scale(x)
y = np.asarray(y, dtype=float)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
xsq = (x ** 2).sum(axis=1)
dist = xsq[:, np.newaxis] - 2 * x.dot(x.T) + xsq[np.newaxis, :]
gamma = 3.0 / np.median(dist)
# gamma = 0.4
print "x.shape: (%d, %d)" % (x.shape[0], x.shape[1])

# C_list = np.array([1e-2, 1e-1, 1, 1e1, 1e2])
C = float(10)
# C_list = np.array([1e-2, 1e-1])
# gamma = 0.1
# rholist = [0.001, 0.005, 0.01, 0.05, 0.1, 1]
rholist = [0.02, 0.05, 0.07, 0.1, 10, ]
n_iter = 10

trainerr_dasvm = {}
testerr_dasvm = {}
obj_dasvm = {}
trainerr_md = {}
testerr_md = {}
obj_md = {}
trainerr_md2 = {}
testerr_md2 = {}
obj_md2 = {}
trainerr_cd = []
testerr_cd = []
obj_cd = []
trainerr_pega = []
testerr_pega = []
obj_pega = []
for i, rho in enumerate(rholist):
    trainerr_dasvm[i] = []
    testerr_dasvm[i] = []
    obj_dasvm[i] = []

    trainerr_md[i] = []
    testerr_md[i] = []
    obj_md[i] = []
    trainerr_md2[i] = []
    testerr_md2[i] = []
    obj_md2[i] = []

b = 1
c = 4
rs = cv.ShuffleSplit(x.shape[0], n_iter=n_iter, train_size=0.4, test_size=0.5, random_state=None)
for k, (train_index, test_index) in enumerate(rs):
    print "iter # %d" % k
    n = train_index.size
    start_t = time.time()
    clf_da = {}
    # for i, rho in enumerate(rholist):
    #     clf_da[i] = DualKSVM(lmda=C / n, kernelstr='rbf', gm=gamma, nsweep=n, b=b, c=c, verbose=True, rho=rho,
    #                       algo_type='scg_da')
    #     clf_da[i].fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
    #     trainerr_dasvm[i].append(clf_da[i].err_tr)
    #     testerr_dasvm[i].append(clf_da[i].err_te)
    #     obj_dasvm[i].append(clf_da[i].obj)
    # print "dualsvm time %f " % (time.time() - start_t)

    clf_md = {}
    for i, rho in enumerate(rholist):
        clf_md[i] = DualKSVM(lmda=C / n, kernelstr='rbf', gm=gamma, nsweep=n, b=b, c=c, verbose=True, rho=rho,
                          algo_type='sbmd')
        clf_md[i].fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
        trainerr_md[i].append(clf_md[i].err_tr)
        testerr_md[i].append(clf_md[i].err_te)
        obj_md[i].append(clf_md[i].obj)
        # trainerr_md2[i].append(clf_md[i].err_tr2)
        # testerr_md2[i].append(clf_md[i].err_te2)
        obj_md2[i].append(clf_md[i].obj2)
    print "dualsvm time %f " % (time.time() - start_t)

    start_t = time.time()
    clf_cd = DualKSVM(lmda=C / n, kernelstr='rbf', gm=gamma, nsweep=int(4 * n), algo_type='cd')
    clf_cd.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
    trainerr_cd.append(clf_cd.err_tr)
    testerr_cd.append(clf_cd.err_te)
    obj_cd.append(clf_cd.obj)
    print "cd svm time %f" % (time.time() - start_t)

    start_t = time.time()
    clf_pega = Pegasos(lmda=C / n, gm=gamma, kernelstr='rbf', nsweep=5, batchsize=1)
    clf_pega.fit(x[train_index, :], y[train_index], x[test_index, :], y[test_index])
    trainerr_pega.append(clf_pega.err_tr)
    testerr_pega.append(clf_pega.err_te)
    obj_pega.append(clf_pega.obj)
    trainerr_pega.append(clf_pega.err_tr)
    testerr_pega.append(clf_pega.err_te)
    obj_pega.append(clf_pega.obj)
    print "pegasos time %f" % (time.time() - start_t)

for i, row in enumerate(rholist):
    trainerr_dasvm[i] = np.asarray(trainerr_dasvm[i])
    testerr_dasvm[i] = np.asarray(testerr_dasvm[i])
    obj_dasvm[i] = np.asarray(obj_dasvm[i])
    trainerr_md[i] = np.asarray(trainerr_md[i])
    testerr_md[i] = np.asarray(testerr_md[i])
    obj_md[i] = np.asarray(obj_md[i])
    # trainerr_md2[i] = np.asarray(trainerr_md2[i])
    # testerr_md2[i] = np.asarray(testerr_md2[i])
    obj_md2[i] = np.asarray(obj_md2[i])

trainerr_cd = np.asarray(trainerr_cd)
testerr_cd = np.asarray(testerr_cd)
obj_cd = np.asarray(obj_cd)
trainerr_pega = np.asarray(trainerr_pega)
testerr_pega = np.asarray(testerr_pega)


# ----------------plot objective-----------------
plt.figure()
# for i, rho in enumerate(rholist):
#     plt.errorbar(clf_da[1].nker_opers, obj_dasvm[i].mean(axis=0), yerr=obj_dasvm[i].std(axis=0),
#                  label=r'cda$\rho=%f$' % rho)
for i, rho in enumerate(rholist):
    plt.errorbar(clf_md[1].nker_opers, obj_md[i].mean(axis=0), yerr=obj_md[i].std(axis=0),
                 label=r'md $\rho=%f$' % rho)
plt.errorbar(clf_cd.nker_opers, obj_cd.mean(axis=0), yerr=obj_cd.std(axis=0), label='cd')
# plt.errorbar(clf4.nker_opers, obj_dasvm2[C].mean(axis=0), yerr=obj_dasvm2[C].std(axis=0), fmt='--', label=r'scg, $\rho=$%2.2f' % rho2)
plt.legend(loc='best')
plt.title(r'C=%2.4f, $\gamma=%2.2f$' % (C, gamma))
plt.xscale('log')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel('number of kernel evaluations')
plt.ylabel('dual objective')
# plt.show()
plt.savefig('svmtest/output/usps_obj_C_%2.2f_gm_%2.2f.eps' % (C, gamma))

plt.figure()
for i, rho in enumerate(rholist):
    plt.errorbar(clf_md[1].nker_opers, obj_md2[i].mean(axis=0), yerr=obj_md2[i].std(axis=0),
                 label=r'md2 $\rho=%f$' % rho)
plt.errorbar(clf_cd.nker_opers, obj_cd.mean(axis=0), yerr=obj_cd.std(axis=0), label='cd')
# plt.errorbar(clf4.nker_opers, obj_dasvm2[C].mean(axis=0), yerr=obj_dasvm2[C].std(axis=0), fmt='--', label=r'scg, $\rho=$%2.2f' % rho2)
plt.legend(loc='best')
plt.title(r'C=%2.4f, $\gamma=%2.2f$' % (C, gamma))
plt.xscale('log')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel('number of kernel evaluations')
plt.ylabel('dual objective')
# plt.show()
plt.savefig('svmtest/output/usps_obj2_C_%2.2f_gm_%2.2f.eps' % (C, gamma))

# -----------------plot test error -------------------------
plt.figure()
# plt.subplot(row,col,2)
# for i, rho in enumerate(rholist):
#     plt.errorbar(clf_da[1].nker_opers, testerr_dasvm[i].mean(axis=0), yerr=testerr_dasvm[i].std(axis=0),
#                  label=r'sda, $\rho=$%f' % rho)
for i, rho in enumerate(rholist):
    plt.errorbar(clf_md[1].nker_opers, testerr_md[i].mean(axis=0), yerr=testerr_md[i].std(axis=0),
                 label=r'md, $\rho=$%f' % rho)
plt.errorbar(clf_cd.nker_opers, testerr_cd.mean(axis=0), yerr=testerr_cd.std(axis=0), fmt='o-', label='cd')
plt.errorbar(clf_pega.nker_opers, testerr_pega.mean(axis=0), yerr=testerr_pega.std(axis=0), fmt='D-', label='Pegasos')
plt.ylabel('test error')
plt.xlabel('number of kernel evaluations')
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join('svmtest', 'output', 'usps_train_error_C_%2.2f_gm_%2.2f.eps'
                         % (C, gamma)))

# --------------------plot train error --------------------------
plt.figure()
# for i, rho in enumerate(rholist):
#     plt.errorbar(clf_da[i].nker_opers, trainerr_dasvm[i].mean(axis=0), yerr=trainerr_dasvm[i].std(axis=0),
#                 label=r'scg')
for i, rho in enumerate(rholist):
    plt.errorbar(clf_md[i].nker_opers, trainerr_md[i].mean(axis=0), yerr=trainerr_md[i].std(axis=0),
                 label=r'md')
plt.errorbar(clf_cd.nker_opers, trainerr_cd.mean(axis=0), yerr=trainerr_cd.std(axis=0), fmt='o-', label='cd-svm')
plt.errorbar(clf_pega.nker_opers, trainerr_pega.mean(axis=0), yerr=trainerr_pega.std(axis=0), fmt='D-',
             label='Pegasos')
plt.ylabel('train error')
# plt.errorbar(clf4.nker_opers, testerr_dasvm2[C].mean(axis=0), yerr=testerr_dasvm2[C].std(axis=0), fmt='s-', label=r'scg, $\rho=$%2.2f' %rho2)
plt.xlabel('number of kernel evaluations')
plt.title(r'C=%2.4f, $\gamma=%2.2f$' % (C, gamma))
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.xlim(xmin=1)
plt.savefig('svmtest/output/usps_test_error%d_%d_C_%2.2f_gm_%2.2f.eps' % (pos_class, neg_class, C, gamma))

# plt.errorbar(clf_da.nker_opers, testerr_dasvm[C].mean(), yerr=testerr_dasvm[C].std(axis=0), 'x-', label='scg')
# plt.semilogy(clf2.nker_opers, testerr_cd[C], 'o-', label='cd')
# plt.semilogy(clf_pega.nker_opers, testerr_pega[C], 'D-', label='Pegasos')


