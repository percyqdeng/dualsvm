__author__ = 'qd'

from distutils.core import setup
from Cython.Build import cythonize
from sklearn._build_utils import get_blas_info
import os
import numpy as np
path = "auxiliary/"
# path = ""
cblas_libs, blas_info = get_blas_info()

tmp = '/opt/local/include'
libraries = []
if os.name == 'posix':
    cblas_libs.append('m')
    libraries.append('m')
extensions = [path+"rand_no_repeat.pyx"]
setup(
    # name='',
    include_dirs=[np.get_include(), tmp],
    ext_modules=cythonize(extensions,
                          # language="c++",
                          # libraries=cblas_libs,
                          )
)
# run setup_aix.py build_ext --inplace
