
from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
import numpy as np
from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *
from sklearn import linear_model
from load_data import load_mnist, convert_binary

data = load_mnist()
pos_ind = 6
neg_ind = 5
x, y = convert_binary(data, pos_ind, neg_ind)

xsum = x.sum(axis=0)
ind = np.where(xsum>0)  # return object is tuple
x = x[:, ind[0]]
x = x.astype(float)
y = y.astype(float)
n, p = x.shape
random_state = 21
lmda = 0.01
nsweep = 1
xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=0.2, random_state=random_state)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
xtrain = min_max_scaler.fit_transform(xtrain)
xtest = min_max_scaler.transform(xtest)
ntrain = ytrain.size

alphas, coefs, gaps = linear_model.lasso_path(xtrain, ytrain,n_alphas=20, return_models=False, fit_intercept=False)
alphas = alphas[::-1]
zs = (coefs==0).sum(axis=0)
zs = zs[::-1]
gaps = gaps[::-1]

zs2 = np.zeros(len(alphas))
obj2 = np.zeros(len(alphas))
zs3 = np.zeros(len(alphas))
obj3 = np.zeros(len(alphas))
for i, alpha in enumerate(alphas):
    print "alpha: %f" % alpha
    clf = LassoLI(lmda=alpha, algo='cd', T=1000)
    clf.fit(xtrain, ytrain)
    zs2[i] = clf.num_zs[-1]
    obj2[i] = clf.train_obj[-1]

    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False)
    clf.fit(xtrain, ytrain)
    zs3[i] = (clf.coef_==0).sum()
    obj3[i] = 0.5/ntrain * ((ytrain-xtrain.dot(clf.coef_))**2).sum() + alpha * np.linalg.norm(clf.coef_, ord=1)
plt.figure()
plt.plot(alphas, zs, 'x-', label='linear_path')
plt.plot(alphas, zs2, 'D-', label='mine')
plt.plot(alphas, zs3, '<-', label='sklearn')

plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('error')
plt.legend(loc='best')
plt.savefig('mnist_cd_path.eps')
plt.figure()
plt.plot(alphas, gaps, 'x-', label='linear_path')
plt.plot(alphas, obj2, 'D-', label='mine')
plt.plot(alphas, obj3, '<-', label='sklearn')
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('number of zeros')
plt.legend(loc='best')