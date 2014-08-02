from sklearn import preprocessing
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
import sklearn.cross_validation as cv
from sklearn.linear_model import SGDClassifier
from load_data import convert_binary, load_usps
from mylasso import *

pos_class = 3
neg_class = 5
random_state = 33
n_iter = 1

data = load_usps()
x, y = convert_binary(data, pos_class, neg_class)
# sss = cv.StratifiedShuffleSplit(y, n_iter=n_iter, test_size=0.4, train_size=0.6, random_state=random_state)
#
# for i, (train_index, test_index) in sss:
xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=0.8, random_state=random_state)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
xtrain = min_max_scaler.fit_transform(xtrain)
xtest = min_max_scaler.transform(xtest)

n, p = xtrain.shape
lmda = 0.01
b = 4
c = 4
nsweep = 10
T1 = nsweep * n
rgr = LassoLI(lmda=lmda, b=b, c=c, T=T1, algo='scg')
rgr.fit(xtrain, ytrain)
rgr.predict(xtest)

T2 = (T1*(b*c)/p)
rgr2 = LassoLI(lmda=lmda, T=T2, algo='rda')
rgr2.fit(xtrain, ytrain)

T3 = (T1*(b*c)/n)
rgr3 = LassoLI(lmda=lmda, T=T3, algo='cd')
rgr3.fit(xtrain, ytrain)


# clf = SGDClassifier(loss="squared_loss", penalty="l1", fit_intercept=False, n_iter=nsweep, alpha=lmda)
# clf.fit(xtrain, ytrain)

plt.figure()
plt.subplot(211)
plt.plot(rgr.num_features, rgr.obj, 'bo-', label='scg')
plt.plot(rgr2.num_features, rgr2.obj, 'gd-', label='rda')
plt.plot(rgr3.num_features, rgr3.obj, 'r>-', label='cd')
plt.xlabel('number of features')
plt.ylabel('train error')
# plt.figure()
plt.subplot(212)
plt.plot(rgr.num_features, rgr.num_zs, 'bo-', label='scg')
plt.plot(rgr2.num_features, rgr2.num_zs, 'gd-', label='rda')
plt.plot(rgr3.num_features, rgr3.num_zs, 'r>-', label='cd')
plt.xlabel('number of features')
plt.ylabel('train error')
plt.ylabel('number of zeros')
plt.legend()
plt.show()
