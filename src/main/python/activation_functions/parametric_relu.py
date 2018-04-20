from . import ActivationFunction


class ParametricReLU(ActivationFunction):
    def __init__(self, parameter):
        super(ParametricReLU, self).__init__()

        assert isinstance(parameter, (int, float, complex))

        self._parameter = parameter

    def function(self, x):
        return x if x > 0 else self._parameter * x

    def derivative(self, x):
        return 1 if x > 0 else self._parameter
