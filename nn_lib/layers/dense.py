from layer import Layer


class Dense(Layer):
    def __input__(self, size):
        super(Dense, self).__init__(size=size)
