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
sig_D = [10, 20, 30, 40, 50, 60, 70, 90, 110, 130, 150]
# sig_D = [130, 160, 190, 230, 270, 320, 350, 380]

# ntrain = ytrain.size

# alphas, coefs, gaps = linear_model.lasso_path(x, y, n_alphas=6, return_models=False, fit_intercept=False)
# lmda_list = alphas[::-1]
lmda = 0.001
n_iter = 15
random_state =  np.random.randint(low=1,high=10000)
ss = cv.StratifiedShuffleSplit(y=y, n_iter=n_iter, test_size=0.5, random_state=random_state)

# scg_obj = np.zeros((n_iter, len(sig_D)))
scg_obj = None
rda_obj = None
rda2_obj = None
cd_obj = None
scg_tm = None
rda_tm =None
rda2_tm =None
scg_zs =None
rda_zs=None
rda2_zs=None
cd_zs =None
b = 4
c = 1
# for j, r in enumerate(sig_D):
#     scg_obj[r] = None
#     rda_obj[r] = None
#     rda2_obj[r] = None

for i, (train_idx, test_idx) in enumerate(ss):
    xtrain = x[train_idx, :]
    ytrain = y[train_idx]
    xvalid = x[test_idx, :]
    yvalid = y[test_idx]
    ntrain = ytrain.size
    num_ftrs = 5 * int((b+c)*ntrain)
    r = LassoLI.tune_rho(sig_D, xvalid, yvalid, T=int(0.2*num_ftrs/(b+c)), lmda=lmda, b=b, c=c, algo='scg')
    rgr = LassoLI(lmda=lmda, b=b, c=c, T=num_ftrs/(b+c), algo='scg', sig_D=r)
    rgr.fit(xtrain, ytrain)
    if scg_obj is None:
        scg_obj = rgr.train_obj
        scg_zs = rgr.num_zs
    else:
        scg_obj = np.vstack((scg_obj, rgr.train_obj))
        scg_zs = np.vstack((scg_zs, rgr.num_zs))
    print 'scg set rho=%d' % r

    r = LassoLI.tune_rho(sig_D, xvalid, yvalid, T=int(0.2*num_ftrs/p), lmda=lmda, algo='rda')
    rda = LassoLI(lmda=lmda, T=num_ftrs/p, algo='rda', sig_D=r)
    rda.fit(xtrain, ytrain)
    if rda_obj is None:
        rda_obj = rda.train_obj
        rda_zs = rda.num_zs
    else:
        rda_obj = np.vstack((rda_obj, rda.train_obj))
        rda_zs = np.vstack((rda_zs, rda.num_zs))
    print 'rda set rho=%d' % r


    r = LassoLI.tune_rho(sig_D, xvalid, yvalid, T=int(0.2*num_ftrs/(b+c)), lmda=lmda, algo='rda2')
    rda2 = LassoLI(lmda=lmda, T=num_ftrs/(b+c), algo='rda2', b=b, c=c, sig_D=r)
    rda2.fit(xtrain, ytrain)
    print 'rda2 set rho=%d' % r
    if rda2_obj is None:
        rda2_obj = rda2.train_obj
        rda2_zs = rda2.num_zs
    else:
        rda2_obj = np.vstack((rda2_obj, rda2.train_obj))
        rda2_zs = np.vstack((rda2_zs, rda2.num_zs))


plt.figure()
key = sig_D[0]
key2 = sig_D[1]
plt.errorbar(rgr.num_features, scg_obj.mean(axis=0), yerr=scg_obj.std(axis=0), fmt='x--', label='scg')
plt.errorbar(rda.num_features, rda_obj.mean(axis=0), yerr=rda_obj.std(axis=0), fmt='o-', label='rda')
plt.errorbar(rda2.num_features, rda2_obj.mean(axis=0), yerr=rda2_obj.std(axis=0), fmt='v-.', label='rda2')
plt.title(r'$\lambda=$%.3f' % lmda)
plt.legend(loc='best')
plt.xlabel('number of features')
plt.ylabel('error')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('../output/minist_err_val_%d_%d_lmda_%.3f.eps' % (pos_ind, neg_ind, lmda))

plt.figure()
key = sig_D[0]
key2 = sig_D[1]
plt.errorbar(rgr.num_features, scg_zs.mean(axis=0), yerr=scg_zs.std(axis=0), fmt='x--', label='scg')
plt.errorbar(rda.num_features, rda_zs.mean(axis=0), yerr=rda_zs.std(axis=0), fmt='o-', label='rda')
plt.errorbar(rda2.num_features, rda2_zs.mean(axis=0), yerr=rda2_zs.std(axis=0), fmt='v-.', label='rda2')
plt.title(r'$\lambda=$%.3f' % lmda)
plt.legend(loc='best')
plt.xlabel('number of features')
plt.ylabel('number of zeros')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('../output/minist_zs_val_%d_%d_lmda_%.3f.eps' % (pos_ind, neg_ind, lmda))

