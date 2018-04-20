import numpy as np

from . import ActivationFunction


class Sigmoid(ActivationFunction):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.function(x) * (1 - self.function(x))
