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
x = x[:, ind[0]]
x = x.astype(float)

random_state = 21

eta_list = [0.1, 0.5, 1, 2, 5, 7]
xtrain, xtest, ytrain, ytest = cv.train_test_split(x, y, test_size=0.2, random_state=random_state)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
xtrain = min_max_scaler.fit_transform(xtrain)
xtest = min_max_scaler.transform(xtest)
ntrain = ytrain.size

lmda = 0.001
b = 4
c = 1
test_obj = np.zeros(len(eta_list))
num_zs = np.zeros(len(eta_list))
for i, eta in enumerate(eta_list):
    nsweep = 10
    T1 = nsweep * ntrain
    rgr = LassoLI(lmda=lmda, b=b, c=c, T=T1, algo='scg', eta=eta)
    rgr.fit(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
    rgr.predict(xtest)
    test_obj[i] = rgr.test_obj[-1]
    num_zs[i] = rgr.num_zs[-1]

plt.figure()
plt.plot(eta_list, test_obj, '-bo')
plt.ylabel('test objective')
plt.figure()
plt.plot(eta_list, num_zs, '-bo')
plt.ylabel('number of zeros')
