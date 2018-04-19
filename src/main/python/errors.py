class BaseNnLibError(Exception):
    pass


class InvalidLayerError(BaseNnLibError):
    pass


class InvalidDataError(BaseNnLibError):
    pass


class InvalidCallError(BaseNnLibError):
    pass
