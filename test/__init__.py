import functools


def supported(lib, version="", reason=""):
    def wrapped(func):
        if not hasattr(func, "library_support"):
            func.library_support = []
        func.library_support.append((lib, version, reason))

        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return wrapped


def unsupported(lib, reason=""):
    def wrapped(func):
        if not hasattr(func, "library_support"):
            func.library_support = []
        func.library_support.append((lib, "unsupported", reason))

        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return wrapped
