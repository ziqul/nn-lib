from ..errors import InvalidDataError


class Layer(object):
    def __init__(self, neurons_amount):
        if type(neurons_amount) != int:
            raise InvalidDataError(
                'Size of layer has to be specified by integer.')

        self._output_size = neurons_amount
        self._input_size = None
        self._learning_rate = None

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def get_output_size(self):
        return self._output_size

    def get_input_size(self):
        return self._input_size
