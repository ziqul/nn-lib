class ActivationFunction():
    def __new__(cls, *args, **kwargs):
        if cls is ActivationFunction:
            raise TypeError(
                'ActivationFunction class may not be instantiated.')

        return object.__new__(cls, *args, **kwargs)
