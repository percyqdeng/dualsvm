# distutils: language = c++
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand

cdef class RandNoRep(object):
    """
    generate random numbers between 0 to n-1 without replacement
    """
    def __init__(self, int n):
        self.seeds = np.zeros(n, dtype=np.intp)
        self.n = n
        cdef Py_ssize_t i
        for i in range(n):
            self.seeds[i] = i

    cdef k_choice(self, Py_ssize_t [::1] arr, Py_ssize_t k):
        """
        return k elements w/o replacement, the amortized complexity is linear.
        """
        # assert (k <= self.n-1)
        # assert (k <=arr.size)
        cdef Py_ssize_t pt = self.n-1
        cdef Py_ssize_t i, j, tmp
        for i in range(k):
            j = rand() % self.n
            tmp = self.seeds[pt]
            self.seeds[pt] = self.seeds[j]
            self.seeds[j] = tmp
            arr[i] = self.seeds[pt]
            pt -= 1