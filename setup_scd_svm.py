__author__ = 'qd'

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from sklearn._build_utils import get_blas_info

"dsvm/scg_svm_on_fly.pyx"
extensions = ["dsvm/dsvm.pyx",  "dsvm/kernel_func.pyx", "dsvm/rand_dataset.pyx"]
setup(
    # name='stoc_coor_cython',
    include_dirs=[np.get_include(), '/opt/local/include/'],
    ext_modules=cythonize(extensions,
                          # language="C++",
                          )
)

# if __name__ == "__main__":

# run setup_scd_svm.py build_ext --inplace