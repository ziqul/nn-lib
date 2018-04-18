from layer import Layer


class Input(Layer):
    def __input__(self, size):
        super(Input, self).__init__(size=size)
