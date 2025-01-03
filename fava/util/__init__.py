import time

from fava.util._mpi import mpi
from fava.util._types import HID_T, NP_T


def timer(func):
    def decorator(*args, **kwargs):
        tbeg = time.perf_counter()
        result = func(*args, **kwargs)
        tend = time.perf_counter()
        if mpi.root:
            print(f"Timing: {func.__name__} --> {tend-tbeg:2.4f}", flush=True)
        return result

    return decorator
