

def train_lasso(double [:,::1] x, int[::1]y, int b=1, int c=2, double lmda=0.1, Py_ssize_t T=100):


def train_test_lasso(double [:,::1] x, int[::1]y, double[:,::1]xtest, int[:]ytest,
                int b=1, int c=2, double lmda=0.1, Py_ssize_t T=100):
    """
    coordinate descent for lasso
    :return:
    """