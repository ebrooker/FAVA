from typing import Any
import numpy as np


class HDF5_TYPES:
    F32: str = "<f4"
    F64: str = "<f8"
    F64_PARAMETER: list[tuple[str]] = [("name", "S256"), ("value", "<f8")]

    I32: str = "<i4"
    I64: str = "<i8"
    I32_PARAMETER: list[tuple[str]] = [("name", "S256"), ("value", "<i4")]

    BOOL_PARAMETER: dict[str, Any] = {
        "names": ["name", "value"],
        "formats": ["S256", "<i4"],
        "offsets": [4, 0],
        "itemsize": 260,
    }

    STR_PARAMETER: dict[str, Any] = {
        "names": ["name", "value"],
        "formats": ["S256", "S256"],
        "offsets": [256, 0],
        "itemsize": 512,
    }

    UNKNOWN_NAMES = "S4"


HID_T = HDF5_TYPES()


class NUMPY_TYPES:
    FLOAT32 = np.float32()
    FLOAT64 = np.float64()
    INT32 = np.int32()
    INT64 = np.int64()


NP_T = NUMPY_TYPES()
