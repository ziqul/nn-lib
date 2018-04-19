import numpy as np

from src.main.python.model import Model
from src.main.python.layers import Input
from src.main.python.layers import Dense


def main():
    input = [0.5]
    expected_result = [1, 0]

    model = Model()
    model.add_layer(Input(1))
    model.add_layer(Dense(2))
    model.build()

    given_result = None

    mse = np.inf
    itern = 0

    while mse > 0.001 and itern < 100000:
        given_result = model.forward_pass(input)
        _given_result = np.array(given_result)
        _expected_result = np.array(expected_result)

        errors = list(_given_result - _expected_result)

        model.backward_pass(errors)

        mse = mean_squared_error(errors)
        itern += 1

    print('Finished on iter', itern)
    print('Last given result', given_result)
    print('MSE', mse)


def mean_squared_error(errors):
    np_errors = np.array(errors)

    sqr_errors = np_errors ** 2

    mse = np.sum(sqr_errors) / len(sqr_errors)

    return mse


if __name__ == '__main__':
    main()
