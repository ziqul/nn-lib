from . import ActivationFunction


class LeakyReLU(ActivationFunction):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def function(self, x):
        return x if x > 0 else 0.01 * x

    def derivative(self, x):
        return 1 if x > 0 else 0.01
