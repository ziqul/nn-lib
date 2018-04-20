from enum import Enum


class GradientDescentType(Enum):
    BATCH = 1
    MINI_BATCH = 2
    STOCHASTIC = 3
