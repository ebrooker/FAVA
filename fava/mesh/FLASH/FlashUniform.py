"""
FLASH Uniform dataset

Includes a `from_amr` constructor

Has methods to write out all pertinent data for an HDF5 file.

However, this should probably be checked against a FLASH generated Uniform grid model

Functions that write the same way as AMR FLASH models should be moved to a parent class

"""

from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from functools import cached_property

import logging
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
import h5py

from fava.mesh.FLASH._flash import FLASH
from fava.mesh.FLASH._util import FIELD_MAPPING, NGUARD, MESH_NDIM
from fava.util import mpi, HID_T, NP_T

logger: logging.Logger = logging.getLogger(__file__)


class FlashUniform(FLASH):

    def __init__(self, filename: Optional[str | Path] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.filename: str | Path | None = filename

    @classmethod
    def is_this_your_mesh(self, filename: str | Path, *args, **kwargs) -> bool:
        fn_types: Tuple[str, str] = ("hdf5_uniform_",)
        return any(fn in filename for fn in fn_types)

    @property
    def filename(self) -> Path:
        return self._filename

    @filename.setter
    def filename(self, filename: str | Path) -> None:

        if not isinstance(filename, (str, Path)):
            msg: str = f"Filename must be passed in as a {str} or {Path}; not {type(filename)}"
            logger.error(msg)
            return

        _fn: Path = Path(filename)

        if _fn == self._filename:
            msg: str = f"File already selected: {filename}"
            return

        self._filename = _fn

    def load_data(self, names: List[str] | None = None) -> None:
        fields = names if names is not None else self.fields
        with h5py.File(name=self.filename, mode="r") as f:
            for field in fields:
                self._read_variable_data(handle=f, name=f"{field:4s}")

    def data(self, name: str) -> NDArray:

        field: str | None = name
        if field not in self.fields:
            field = FIELD_MAPPING.get(name)

        if field is None:
            logger.warning("Cannot find %s in dataset", name)
            return

        if field not in self._data:
            with h5py.File(name=self.filename, mode="r") as f:
                self._read_variable_data(handle=f, name=f"{field:4s}")

        return self._data[field]

    def load(self) -> None:
        """
        This function loads in all FLASH file data, except for the UNK arrays.

        I.e., information pertaining to runtime/scalars, block information, etc
        are all loaded into memory. This data should not be relatively small when
        compared to loading in UNK arrays. We allow a user to choose what fields
        in UNK get loaded in to control memory usage.

        DEV NOTES 2024/11/24: We may move to using more just-in-time reads for some parts of
        the data loaded in, such as coordinates, block bounds, etc if any memory
        limitations begin to appear
        """
        if not self.filename.is_file():
            msg: str = f"File does not exist: {self.filename}"
            logger.error(msg)
            return

        try:

            # Delete loaded variable data and any cached properties
            self._data: dict[str, NDArray] = {}
            self._delete_cached_properties()

            # Open the file in context to ensure strong closing upon success/failure
            with h5py.File(name=self.filename, mode="r") as f:
                self._read_scalars(handle=f)
                self._read_runtime_parameters(handle=f)

                try:
                    self._set_integers()
                    self._set_reals()
                except Exception as exc:
                    logger.exception(
                        "Error setting attributes from FLASH FILE %s", self.filename, exc_info=True
                    )
                    raise RuntimeError from exc

                self._read_nvar_names(handle=f)
                self._read_coordinates(handle=f)
                self._read_block_sizes(handle=f)
                self._read_bounds(handle=f)
                self._read_refine_level(handle=f)

        except Exception as exc:
            logger.exception("Error reading FLASH FILE %s", self.filename, exc_info=True)
            raise RuntimeError from exc
