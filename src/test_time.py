__author__ = 'qdengpercy'


import numpy as np
import time
import timeit
import cmp_time_usps


def profile_gen_rand():
    import pstats
    import cProfile
    import pyximport
    pyximport.install()
    n = 19755634
    fun_call = "cmp_time.gen_rand_int2(n)"
    start_time = time.time()
    cmp_time_usps.gen_rand_int(n)
    t1 = time.time() - start_time

    start_time = time.time()
    cmp_time_usps.gen_rand_int2(n)
    t2 = time.time() - start_time
    print t1, t2


def cmp_time_mat_access():
    n = 10000
    start_time = time.time()
    cmp_time_usps.mat_access(n)
    t1 = time.time() - start_time

    start_time = time.time()
    cmp_time_usps.mat_access2(n)
    t2 = time.time() - start_time

    start_time = time.time()
    cmp_time_usps.mat_access3(n)
    t3 = time.time() - start_time

    start_time = time.time()
    cmp_time_usps.mat_access4(n)
    t4 = time.time() - start_time
    print t1, t2, t3, t4

def cmp_perm_uniform():
    # compute time of permutation and uniform without replacement
    start = time.time()
    n = 100000
    sample = np.random.permutation(n)
    cost1 = time.time() - start

    start = time.time()
    sample = np.random.choice(n, n, replace=False)
    cost2 = time.time() - start

    print "time cost"
    print "rand perm "+str(cost1)
    print "rand int w/o replace "+str(cost2)

def cmp_for_loop():
    n = 100000
    a = np.random.normal(0,1,n)
    b = np.random.normal(0, 1, n)

    start = time.time()
    res1 = np.dot(a, b)
    t1 = time.time() - start

    start = time.time()
    res2 = 0
    for i in range(n):
        res2 += a[i] + b[i]
    t2 = time.time() - start
    print "time1 " + str(t1)
    print "time2 " + str(t2)

if __name__ =="__main__":
    # cmp_for_loop()
    # profile_gen_rand()
    cmp_time_mat_access()