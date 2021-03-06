__author__ = 'qd'

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
path = "online_lasso/"
# path = ""
extensions = [path+"scg_lasso.pyx", path+"rda_lasso.pyx"]
setup(
    # name='',
    include_dirs=[np.get_include(), '.'],
    ext_modules=cythonize(extensions,
                          # language="c++",
                          libraries=['cblas', 'lapack'],
                          )
)

# if __name__ == "__main__":
#     os.

# run setup_lasso.py build_ext --inplace