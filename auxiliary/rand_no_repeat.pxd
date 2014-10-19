# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np

cdef class RandNoRep(object):
    """
    generate random numbers between 0 to n-1 without replacement
    """
    cdef Py_ssize_t [::1] seeds
    cdef Py_ssize_t n
    cdef Py_ssize_t b_size # batch size
    cdef k_choice(self, Py_ssize_t [::1] arr, Py_ssize_t k)

