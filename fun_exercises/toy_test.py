import numpy as np
import time
from cmp_cblas import *
# import cmp_cblas
# N = 1000000
# a = np.random.normal(0, 1, N)
# b = np.random.normal(0, 1, N)
#
# start_t = time.time()
# c = ddot_blas(a, b)
# print "time %f, a.T*b=%f" % (time.time()-start_t, c)
# start_t = time.time()
# c = ddot_cy(a, b)
# print "time %f, a.T*b=%f" % (time.time()-start_t, c)

n = 10

aa = np.random.normal(0,1,(n, n))
b = np.random.normal(0,1,n)

c = mat_vec_blas(aa, b)


c = mat_vec_cy(aa, b)