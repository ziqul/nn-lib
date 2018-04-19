from . import Layer


class Input(Layer):
    def __init__(self, neurons_amount):
        super(Input, self).__init__(neurons_amount=neurons_amount)

        self._input_size = neurons_amount

    def pass_input(self, input):
        self._input = input

    def get_output(self):
        return self._input

    def pass_error(self, error):
        pass
