import time

from fava.util._mpi import mpi
from fava.util._types import HID_T, NP_T


def timer(func):
    def decorator(*args, **kwargs):
        tbeg = time.perf_counter()
        result = func(*args, **kwargs)
        tend = time.perf_counter()
        print(f"Timing: {func.__name__} --> {tend-tbeg:2.4f}")
        return result

    return decorator
