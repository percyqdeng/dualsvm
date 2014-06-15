
import os

if os.name == "nt":
     dtpath = "..\\..\\dataset\\ucibenchmark\\"
elif os.name == "posix":
    dtpath = '../dataset/'

i = 2
trainfile = dtpath+'usps/usps.' + str(i)
testfile = dtpath+'usps/usps.t.' + str(i)
modelfile = 'lasvm-source/model/model_train_'+str(i)
traincmd = "./lasvm-source/la_svm " + trainfile + " " + modelfile
os.system(traincmd)
testcmd = './lasvm-source/la_test '+'-B 0 ' + testfile + ' ' + modelfile + ' laoutput'
os.system(testcmd)