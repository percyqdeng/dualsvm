from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *
from load_data import load_mnist, convert_binary



data = load_mnist()
pos_ind = 3
neg_ind = 5

x, y = convert_binary(data, pos_ind, neg_ind)

xsum = x.sum(axis=0)
ind = np.where(xsum>0)  # return object is tuple
x = x.astype(float)
x = x[:, ind[0]]
n, p = x.shape
random_state = 21
lmda = 0.01
nsweep = 1
xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=0.4, random_state=random_state)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
xtrain = min_max_scaler.fit_transform(xtrain)
xtest = min_max_scaler.transform(xtest)
ntrain = ytrain.size

b = 4
c = 4


# fix number of examples
nsweep = 2
T1 = nsweep * ntrain
rgr = LassoLI(lmda=lmda, b=b, c=c, T=T1, algo='scg')
rgr.fit(xtrain, ytrain)
rgr.predict(xtest)

T2 = nsweep * ntrain
rgr2 = LassoLI(lmda=lmda, T=T2, algo='rda')
rgr2.fit(xtrain, ytrain)

T3 = nsweep * p
rgr3 = LassoLI(lmda=lmda, T=T3, algo='cd')
rgr3.fit(xtrain, ytrain)

plt.figure()
plt.subplot(211)
plt.plot(rgr.num_iters, rgr.obj, 'bo-', label='scg')
plt.plot(rgr2.num_iters, rgr2.obj, 'gd-', label='rda')
plt.plot(rgr3.num_iters, rgr3.obj, 'r>-', label='cd')
plt.xlabel('number of features')
plt.ylabel('train error')
# plt.figure()
plt.subplot(212)
plt.plot(rgr.num_iters, rgr.num_zs, 'bo-', label='scg')
plt.plot(rgr2.num_iters, rgr2.num_zs, 'gd-', label='rda')
plt.plot(rgr3.num_iters, rgr3.num_zs, 'r>-', label='cd')
plt.xlabel('number of iterations')
plt.ylabel('train error')
plt.ylabel('number of zeros')
plt.legend()
plt.show()

# clf = SGDClassifier(loss="squared_loss", penalty="l1", fit_intercept=False, n_iter=nsweep, alpha=lmda)
# clf.fit(xtrain, ytrain)

# plt.figure()
# plt.subplot(211)
# plt.plot(rgr.num_features, rgr.obj, 'bo-', label='scg')
# plt.plot(rgr2.num_features, rgr2.obj, 'gd-', label='rda')
# plt.plot(rgr3.num_features, rgr3.obj, 'r>-', label='cd')
# plt.xlabel('number of features')
# plt.ylabel('train error')
# # plt.figure()
# plt.subplot(212)
# plt.plot(rgr.num_features, rgr.num_zs, 'bo-', label='scg')
# plt.plot(rgr2.num_features, rgr2.num_zs, 'gd-', label='rda')
# plt.plot(rgr3.num_features, rgr3.num_zs, 'r>-', label='cd')
# plt.xlabel('number of features')
# plt.ylabel('train error')
# plt.ylabel('number of zeros')
# plt.legend()
# plt.show()



