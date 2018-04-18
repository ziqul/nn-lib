from .errors import InvalidLayerError
from .errors import InvalidDataError
from .errors import InvalidCallError
from .layers import Input
from .layers import Layer

import numpy as np


class Model():
    def __init__(self):
        self._layers = []
        self._built = False

    def add_layer(self, layer):
        if type(layer) != Layer:
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
        if not self._built:
            raise InvalidCallError(
                'Model.forward_pass() cannot be '
                'called until Model.build() is called.')

        if type(input) != np.array:
            raise InvalidDataError(
                'Invalid input type. '
                'Model.forward_pass() requires '
                '{required_type} as input. '
                '{given_type} is given instead.'
                .format(
                    required_type=np.array,
                    given_type=type(input)))

        if input.shape[0] != self._layers[0].size():
            raise InvalidDataError(
                'Invalid size of input. '
                'Input layer of this model requires array of size'
                '{required_size} as input. '
                'Array of {given_size} is given instead.'
                .format(
                    required_size=self._layers[0].size(),
                    given_size=input.shape[0]))

        pass

    def backward_pass(self, error):
        if not self._built:
            raise InvalidCallError(
                'Model.backward_pass() cannot be '
                'called until Model.build() is called.')

        if type(error) != np.array:
            raise InvalidDataError(
                'Invalid input type. '
                'Model.backward_pass() requires '
                '{required_type} as input. '
                '{given_type} is given instead.'
                .format(
                    required_type=np.array,
                    given_type=type(error)))

        if error.shape[0] != self._layers[len(self._layers) - 1].size():
            raise InvalidDataError(
                'Invalid size of input. '
                'Input layer of this model requires array of size'
                '{required_size} as input. '
                'Array of {given_size} is given instead.'
                .format(
                    required_size=self._layers[len(self._layers) - 1].size(),
                    given_size=error.shape[0]))

        pass

    def build(self):
        weights = []

        for i, layer in enumerate(self._layers):
            if i == 0:
                continue

            weights.append(
                np.random.rand(
                    self._layers[i - 1].size(),
                    self._layers[i].size()
                )
            )

        self._weights = np.array(weights)
        self._built = True
