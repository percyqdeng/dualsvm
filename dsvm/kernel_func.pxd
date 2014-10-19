# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
cdef enum kernel_type_t:
    RBF = 1
    POLY = 2
ctypedef kernel_type_t kernel_type

cdef class KernelFunc(object):
    cdef int ktype
    cdef double gamma
    cdef int degree

    cdef double product(self, double[::1]x1, double[::1] x2)
    # cpdef kernel_matrix(self, x1, x2)

