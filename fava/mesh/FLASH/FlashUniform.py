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
from math import log2
import logging
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
import h5py
import itertools
from fava.mesh.FLASH._flash import FLASH
from fava.mesh.FLASH._util import FIELD_MAPPING, NGUARD, MESH_MDIM
from fava.util import mpi, HID_T, NP_T
from scipy.stats import binned_statistic

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

    def fractal_dimension(self, field: str, contour: float) -> dict:

        height, width, depth = self.nCellsVec

        self.load_data(names=[field])

        if contour is None:
            _contour: float = self._data[field].mean()
        else:
            _contour: float = contour

        edata: NDArray = np.zeros((height, width, depth), dtype=np.int8)
        edata[self._data[field] == _contour] = 1

        depth_start: int = 0
        if depth != 1:
            depth_start = 1
        else:
            depth += 1

        _data: dict = {"contour": _contour}

        iterations: int = int((height - 2) * (width - 2) * (depth - depth_start - 1))
        lb, ub = mpi.parallel_range(iterations=iterations)

        for i, j, k in list(
            itertools.product(range(1, height - 1), range(1, width - 1), range(depth_start, depth - 1))
        )[lb:ub]:
            val = self._data[field][i, j, k]

            if val < _contour:
                hidx = _contour - val

                if self._data[field][i + 1, j, k] > _contour:
                    if int(hidx / (self._data[field][i + 1, j, k] - val)) == 0:
                        edata[i, j, k] = 1
                    else:
                        edata[i + 1, j, k] = 1

                if self._data[field][i, j + 1, k] > _contour:
                    if int(hidx / (self._data[field][i, j + 1, k] - val)) == 0:
                        edata[i, j, k] = 1
                    else:
                        edata[i, j + 1, k] = 1

                if self._data[field][i, j - 1, k] > _contour:
                    if int(hidx / (self._data[field][i, j - 1, k] - val)) == 0:
                        edata[i, j, k] = 1
                    else:
                        edata[i, j - 1, k] = 1

                if self._data[field][i - 1, j, k] > _contour:
                    if int(hidx / (self._data[field][i - 1, j, k] - val)) == 0:
                        edata[i, j, k] = 1
                    else:
                        edata[i - 1, j, k] = 1

                if self._data[field][i, j, k + 1] > _contour:
                    if int(hidx / (self._data[field][i, j, k + 1] - val)) == 0:
                        edata[i, j, k] = 1
                    else:
                        edata[i, j, k + 1] = 1

                if self._data[field][i, j, k - 1] > _contour:
                    if int(hidx / (self._data[field][i, j, k - 1] - val)) == 0:
                        edata[i, j, k] = 1
                    else:
                        edata[i, j, k - 1] = 1

        global_edata: NDArray = np.zeros_like(edata)
        mpi.comm.Allreduce(edata, global_edata, MPI.MAX)

        mpi.comm.barrier()
        lowest_level: int = 0
        largest_dim: int = min(height, width)
        if depth > 1:
            largest_dim = min(largest_dim, depth)

        flength: int = int(log2(largest_dim) - lowest_level + 1)

        result: NDArray = np.zeros((flength, 2))

        for level in range(lowest_level, flength + lowest_level):
            bdim = bdim_k = int(2**level)

            if depth == 1:
                bdim_k = 1

            nfilled: int = 0

            iterations: int = int(
                len(range(0, height, bdim)) * len(range(0, width, bdim)) * len(range(0, depth, bdim_k))
            )
            lb, ub = mpi.parallel_range(iterations=iterations)

            mpi.comm.barrier()
            for i, j, k in list(
                itertools.product(range(0, height, bdim), range(0, width, bdim), range(0, depth, bdim_k))
            )[lb:ub]:
                # bsum = 0
                for bx, by, bz in itertools.product(
                    range(i, i + bdim), range(j, j + bdim), range(k, k + bdim_k)
                ):
                    if global_edata[bx, by, bz] > 0:
                        nfilled += 1
                        break
                    # bsum += edata[bx, by, bz]

                # if bsum > 0:
                #     nfilled += 1
            nfilled_global = mpi.comm.allreduce(nfilled, op=MPI.SUM)
            result[level, 0] = flength - level - 1
            result[level, 1] = np.log2(nfilled_global)

        filled_boxes = 2 ** result[:, 1]
        cum_frac_dim: float = np.sum(np.log2(filled_boxes[:-1] / filled_boxes[1:]))
        avg_frac_dim: float = cum_frac_dim / (filled_boxes.size - 1.0)

        mean: NDArray = np.mean(result, axis=0)
        std: NDArray = np.std(result, axis=0)
        rval: float = np.sum((result[:, 0] - mean[0]) * (result[:, 1] - mean[1])) / (
            np.prod(std) * result.shape[0]
        )
        slope: float = rval * std[1] / std[0]
        regress: NDArray = np.array([slope, rval**2, mean[1] - slope * mean[0]])

        _data["average fractal dimension"] = avg_frac_dim
        _data["slope"] = regress[0]
        _data["R2"] = regress[1]
        _data["curve"] = regress[2]

        return _data

    def kinetic_energy_spectra(self):

        # Get the number of velocity components
        velocity: list[str] = ["velx", "vely", "velz"][: self.ndim]

        k_num = self.nCellsVec[: self.ndim]

        k_start: NDArray = -k_num // 2
        k_end = -k_start - 1

        # Create the k-wavenumber grid for n-dimensions
        k: NDArray = np.array(
            np.meshgrid(
                *(np.linspace(ks, ke, n) for ks, ke, n in zip(k_start, k_end, k_num)),
                indexing="ij",
            )
        )

        # Obtain length of k-vector
        if self.ndim == 1:
            k_abs: NDArray = np.abs(k)
        else:
            k_abs = np.sqrt((k**2).sum(axis=0))

        bins: NDArray = np.arange(np.max(k_num) // 2) - 0.5

        ffts = []

        # For each velocity component, compute the FFT and shift center to k=0
        dens = np.sqrt(self.data("dens"))
        for component in velocity:
            fft: NDArray[np.complex128] = np.fft.fftn(dens * self.data(component), norm="forward")
            fft = np.fft.fftshift(fft)
            ffts.append(fft)
        ffts: NDArray = np.array(ffts)

        power: Dict[str, NDArray] = {"total": 0.5 * (np.abs(ffts) ** 2).sum(axis=0)}

        power["longitudinal"] = np.zeros(k_num, dtype=np.complex128)
        if self.ndim == 1:
            power["longitudinal"] += k * ffts[0, ...]  # x-component velocity for 1D case

        if self.ndim > 1:
            for n in range(self.ndim):
                power["longitudinal"] += k[n] * ffts[n, ...].T

        power["longitudinal"] = np.abs(power["longitudinal"] / np.maximum(k_abs, 1e-99)) ** 2
        power["transverse"] = power["total"] - power["longitudinal"]

        spectral: dict[str, NDArray] = {}
        for key, val in power.items():
            binstats = binned_statistic(k_abs.flatten(), val.flatten(), bins=bins, statistic="mean")

            if "k" not in spectral:
                spectral["k"] = binstats.bin_edges[:-1] + 0.5

            spectral[key] = binstats.statistic

        integral_factor: NDArray = spectral["k"] ** (self.ndim - 1)
        if self.ndim > 1:
            integral_factor *= 2 * np.pi * (self.ndim - 1)

        for key in spectral.keys():
            if key == "k":
                continue
            spectral[key] *= integral_factor

        return spectral
