from .errors import InvalidLayerError
from .errors import InvalidDataError
from .errors import InvalidCallError
from .layers import Input
from .layers import Layer


class Model():
    def __init__(self, learning_rate=0.01):
        self._layers = []
        self._built = False
        self._learning_rate = learning_rate

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise InvalidDataError(
                'Invalid input type. '
                'Model.add_layer() requires '
                '{required_type} as input. '
                '{given_type} is given instead.'
                .format(
                    required_type=Layer,
                    given_type=type(layer)))

        if len(self._layers) == 0 and type(layer) != Input:
            raise InvalidLayerError(
                'First layer in model has to be Input layer.')

        self._layers.append(layer)

    def forward_pass(self, input):
        first_layer = self._layers[0]
        last_layer = self._layers[len(self._layers) - 1]

        if not self._built:
            raise InvalidCallError(
                'Model.forward_pass() cannot be '
                'called until Model.build() is called.')

        if type(input) != list and type(input) != tuple:
            raise InvalidDataError(
                'Invalid input type. '
                'Model.forward_pass() requires '
                '{required_type} as input. '
                '{given_type} is given instead.'
                .format(
                    required_type=', or '.join([str(list), str(tuple)]),
                    given_type=type(input)))

        if len(input) != first_layer.get_input_size():
            raise InvalidDataError(
                'Invalid size of input. '
                'Input layer of this model requires array of size'
                '{required_size} as input. '
                'Array of {given_size} is given instead.'
                .format(
                    required_size=first_layer.get_input_size(),
                    given_size=len(input)))

        first_layer.pass_input(input)

        for i, layer in enumerate(self._layers):
            current_layer = layer

            if current_layer is last_layer:
                break

            next_layer = self._layers[i + 1]

            next_layer.pass_input(current_layer.get_output())

        return last_layer.get_output()

    def backward_pass(self, error):
        first_layer = self._layers[0]
        last_layer = self._layers[len(self._layers) - 1]

        if not self._built:
            raise InvalidCallError(
                'Model.backward_pass() cannot be '
                'called until Model.build() is called.')

        if type(error) != list and type(error) != tuple:
            raise InvalidDataError(
                'Invalid input type. '
                'Model.backward_pass() requires '
                '{required_type} as input. '
                '{given_type} is given instead.'
                .format(
                    required_type=', or '.join([list, tuple]),
                    given_type=type(error)))

        if len(error) != last_layer.get_output_size():
            raise InvalidDataError(
                'Invalid size of input. '
                'Input layer of this model requires array of size'
                '{required_size} as input. '
                'Array of {given_size} is given instead.'
                .format(
                    required_size=last_layer.get_output_size(),
                    given_size=len(error)))

        last_layer.pass_error(error)

        reversed_layers = list(reversed(self._layers))

        for i, layer in enumerate(reversed_layers):
            current_layer = layer

            if current_layer is first_layer:
                break

            prev_layer = reversed_layers[i + 1]

            prev_layer.pass_error(current_layer.get_error())

    def build(self):
        first_layer = self._layers[0]
        last_layer = self._layers[len(self._layers) - 1]

        for i, layer in enumerate(self._layers):
            current_layer = layer

            current_layer.set_learning_rate(self._learning_rate)

            if current_layer is last_layer:
                break

            next_layer = self._layers[i + 1]

            next_layer.set_input_size(current_layer.get_output_size())

        self._built = True
