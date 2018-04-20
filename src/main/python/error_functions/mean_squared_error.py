import numpy as np

from . import ErrorFunction


class MSE(ErrorFunction):
    def __init__(self):
        super(MSE, self).__init__()

    def function(self, errors):
        assert isinstance(errors, (list, tuple))

        np_errors = np.array(errors)

        sqr_errors = np_errors ** 2

        mse = np.sum(sqr_errors) / len(sqr_errors)

        return mse
