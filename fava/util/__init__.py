

import time


def timer(func):
    def decorator(*args, **kwargs):
        tbeg = time.perf_counter()
        result = func(*args, **kwargs)
        tend = time.perf_counter()
        print(f"Timing: {func.__name__} --> {tend-tbeg:2.4f}")
        return result
    return decorator