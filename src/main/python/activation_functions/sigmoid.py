import math

from . import ActivationFunction


class Sigmoid(ActivationFunction):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def function(self, x):
        return 1 / (1 + math.exp(-x))

    def derivative(self, x):
        return self.function(x) * (1 - self.function(x))
