class ErrorFunction():
    def __new__(cls, *args, **kwargs):
        if cls is ErrorFunction:
            raise TypeError(
                'ErrorFunction class may not be instantiated.')

        return object.__new__(cls, *args, **kwargs)
