import numpy as np

from . import Layer
from ..activation_functions import Sigmoid


class Dense(Layer):
    def __init__(self, neurons_amount, activation_function=Sigmoid()):
        super(Dense, self).__init__(neurons_amount=neurons_amount)

        self._activation_function = activation_function

    def set_input_size(self, input_size):
        self._input_size = input_size

        # np.random.rand() returns values in
        # interval [0.0, 1.0), so to achieve
        # distribution in [-0.5, 0.5) we need to
        # detract 0.5 from returned values
        self._weights = \
            np.random.rand(self._input_size, self._output_size) - 0.5

    def pass_input(self, input):
        input = np.array(input)

        vectorized_activation_function = \
            np.vectorize(self._activation_function.function)

        self._prefunction_output = np.dot(input, self._weights)

        self._output = \
            vectorized_activation_function(self._prefunction_output)

    def get_output(self):
        return list(self._output)

    # !!! error has to be returned_output - expected_output !!!
    def pass_error(self, error):
        error = np.array(error)

        vectorized_activation_derivative = \
            np.vectorize(self._activation_function.derivative)

        gradient = \
            vectorized_activation_derivative(self._prefunction_output) * \
            error

        for j in range(0, len(gradient)):
            for i in range(0, len(self._weights[:, j])):
                self._weights[i, j] -= gradient[j] * self._learning_rate

        self._error = \
            np.dot(error, np.ones((self._output_size, self._input_size)))

    def get_error(self):
        return list(self._error)
