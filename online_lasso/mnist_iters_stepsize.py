from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn import linear_model

from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *
from load_data import load_mnist, convert_binary


"""
experiments on relation between convergence and learning rate
"""

data = load_mnist()
pos_ind = 6
neg_ind = 5
x, y = convert_binary(data, pos_ind, neg_ind)
n, p = x.shape
x = x.astype(float)
y = y.astype(float)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x = min_max_scaler.fit_transform(x)
# xtest = min_max_scaler.transform(x)
sig_D = [80,  130, 170, 250, 400]
# ntrain = ytrain.size

# alphas, coefs, gaps = linear_model.lasso_path(x, y, n_alphas=6, return_models=False, fit_intercept=False)
# lmda_list = alphas[::-1]
lmda = 0.01
n_iter = 30
ss = cv.StratifiedShuffleSplit(y=y, n_iter=n_iter, test_size=0.3, random_state=11)

# scg_obj = np.zeros((n_iter, len(sig_D)))
scg_obj = {}
rda_obj ={}
rda2_obj ={}
cd_obj = {}
scg_zs = {}
rda_zs = {}
rda2_zs = {}
cd_zs = {}
num_iters = n * 5
for j, r in enumerate(sig_D):
    scg_obj[r] = None
    rda_obj[r] = None
    rda2_obj[r] = None

for i, (train_idx, test_idx) in enumerate(ss):
    xtrain = x[train_idx, :]
    ytrain = y[train_idx]
    ntrain = ytrain.size
    for r in sig_D:
        b = 4
        c = 1
        rgr = LassoLI(lmda=lmda, b=b, c=c, T=num_iters, algo='scg', sig_D=r)
        rgr.fit(xtrain, ytrain)
        # scg_obj[i, j] = rgr.train_obj[-1]
        if scg_obj[r] is None:
            scg_obj[r] = rgr.train_obj
            scg_zs[r] = rgr.num_zs
        else:
            scg_obj[r] = np.vstack((scg_obj[r], rgr.train_obj))
            scg_zs[r] = np.vstack((scg_zs[r], rgr.num_zs))

        rda = LassoLI(lmda=lmda, T=num_iters, algo='rda', sig_D=r)
        rda.fit(xtrain, ytrain)
        if rda_obj[r] is None:
            rda_obj[r] = rda.train_obj
            rda_zs[r] = rda.num_zs
        else:
            rda_obj[r] = np.vstack((rda_obj[r], rda.train_obj))
            rda_zs[r] = np.vstack((rda_zs[r], rda.num_zs))

        rda2 = LassoLI(lmda=lmda, T=num_iters, algo='rda2', b=b, c=c, sig_D=r)
        rda2.fit(xtrain, ytrain)
        if rda2_obj[r] is None:
            rda2_obj[r] = rda2.train_obj
            rda2_zs[r] = rda2.num_zs
        else:
            rda2_obj[r] = np.vstack((rda2_obj[r], rda2.train_obj))
            rda2_zs[r] = np.vstack((rda2_zs[r], rda2.num_zs))

        cd = LassoLI(lmda=lmda, T=100*p, algo='cd')
        cd.fit(xtrain, ytrain)
        if cd_zs[r] is None:
            cd_zs[r] = cd.num_zs
            cd_obj[r] = cd.train_obj
        else:
            cd_zs[r] = np.vstack((cd_zs[r],  cd.num_zs))
            cd_obj[r] = np.vstack((cd_obj[r],  cd.train_obj))
row = 3
col = 2
ymin = 0.3
ymax = 0.55
plt.subplot(row, col, 2)
for key, value in scg_zs.iteritems():
    plt.errorbar(rgr.num_features, value.mean(axis=0), yerr=value.std(axis=0), label=r'$\rho=$%d' % key)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlim(xmin=0)
plt.subplot(row, col, 4)
for key, value in rda_zs.iteritems():
    plt.errorbar(rda.num_features, value.mean(axis=0), yerr=value.std(axis=0), label=r'$\rho=$%d' % key)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.subplot(row, col, 6)
for key, value in rda2_zs.iteritems():
    plt.errorbar(rda2.num_features, value.mean(axis=0), yerr=value.std(axis=0), label=r'$\rho=$%d' % key)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
# plt.ylim((ymin, ymax))
# plt.legend(bbox_to_anchor=(0, 2, 2.2, 1.1), loc=2, ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('../output/mnist_stepsize.eps')

plt.figure()
key = sig_D[0]
key2 = sig_D[1]
plt.errorbar(rgr.num_features, scg_obj[key].mean(axis=0), yerr=scg_obj[key].std(axis=0), fmt='x--', label='scg')
plt.errorbar(rda.num_features, rda_obj[key].mean(axis=0), yerr=rda_obj[key].std(axis=0), fmt='o-', label='rda')
plt.errorbar(rda2.num_features, rda2_obj[key].mean(axis=0), yerr=rda2_obj[key2].std(axis=0), fmt='v-.', label='rda2')
plt.legend(loc='best')
plt.xlabel('number of features')
plt.ylabel('error')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('../output/minist_error_%d_%d.eps' % (pos_ind, neg_ind))

plt.figure()
key = sig_D[0]
key2 = sig_D[1]
plt.errorbar(rgr.num_features, scg_zs[key].mean(axis=0), yerr=scg_zs[key].std(axis=0), fmt='x--', label='scg')
plt.errorbar(rda.num_features, rda_zs[key].mean(axis=0), yerr=rda_zs[key].std(axis=0), fmt='o-', label='rda')
plt.errorbar(rda2.num_features, rda2_zs[key].mean(axis=0), yerr=rda2_zs[key2].std(axis=0), fmt='v-.', label='rda2')
plt.errorbar(cd.num_features, cd_zs[key].mean(axis=0), yerr=cd_zs[key2].std(axis=0), fmt='v-.', label='rda2')

plt.legend(loc='best')
plt.xlabel('number of samples')
plt.ylabel('number of zeros')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('../output/minist_zs_%d_%d.eps' % (pos_ind, neg_ind))
