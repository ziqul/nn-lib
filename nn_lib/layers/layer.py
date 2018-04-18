from ..errors import InvalidDataError


class Layer(object):
    def __init__(self, size):
        if type(size) != int:
            raise InvalidDataError(
                'Size of layer has to be specified by integer.')

        self._size = size

    def __new__(cls, *args, **kwargs):
        if cls is Layer:
            raise TypeError('Layer class may not be instantiated.')

        return object.__new__(cls, *args, **kwargs)

    def size(self):
        return self._size
