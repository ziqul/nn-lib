import numpy as np

from . import ActivationFunction


class ReLU(ActivationFunction):
    def __init__(self):
        super(ReLU, self).__init__()

    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return 1 if x > 0 else 0
