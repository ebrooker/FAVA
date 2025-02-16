"""
FLASH Uniform dataset

Has methods to write out all pertinent data for an HDF5 file.

However, this should be checked against a FLASH generated Uniform grid model

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
        self.filename = filename

    @classmethod
    def is_this_your_mesh(self, filename: str | Path, *args, **kwargs) -> bool:
        fn_types: Tuple[str, str] = ("hdf5_uniform_",)
        return any(fn in filename for fn in fn_types)

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

    def fractal_dimension(self, field: str, contours: list[float] | float = 0.5) -> dict:

        if isinstance(contours, float):
            _contours: list[float] = [contours]

        height, width, depth = self.nCellsVec

        self.load_data(names=[field])

        retval: dict = {}

        for contour in _contours:

            contour_key: str = f"{contour}"
            retval[contour_key] = {}

            if contour is None:
                _contour: float = self._data[field].mean()
            else:
                _contour: float = contour

            shape: tuple[int] = (height, width, depth)
            dtype = np.int8()
            win: None | MPI.Win = mpi.reallocate(
                id="edata",
                nbytes=dtype.itemsize * np.prod(shape),
                itemsize=dtype.itemsize,
            )
            buffer: MPI.buffer = win.Shared_query(0)[0]

            # Initialize the numpy array with float64 buffer
            edata = np.ndarray(buffer=buffer, dtype=dtype, shape=shape)

            edata[...] = 0
            edata[self._data[field] == _contour] = 1

            depth_start: int = 0
            if depth != 1:
                depth_start = 1
            else:
                depth += 1

            lb, ub = mpi.parallel_range(iterations=height - 2)

            mpi.comm.barrier()

            for i in list(range(1, height - 1))[lb:ub]:
                for j, k in itertools.product(range(1, width - 1), range(depth_start, depth - 1)):

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

                lb, ub = mpi.parallel_range(iterations=len(range(0, height, bdim)))

                for i in list(range(0, height, bdim))[lb:ub]:
                    for j, k in itertools.product(range(0, width, bdim), range(0, depth, bdim_k)):
                        for bx, by, bz in itertools.product(
                            range(i, i + bdim), range(j, j + bdim), range(k, k + bdim_k)
                        ):
                            if edata[bx, by, bz] > 0:
                                nfilled += 1
                                break

                nfilled_global: int = mpi.comm.allreduce(nfilled, op=MPI.SUM)
                result[level, 0] = flength - level - 1
                result[level, 1] = np.log2(nfilled_global)

            mpi.deallocate(id="indices")
            mpi.deallocate(id="edata")
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

            retval[contour_key]["average fractal dimension"] = avg_frac_dim
            retval[contour_key]["slope"] = regress[0]
            retval[contour_key]["R2"] = regress[1]
            retval[contour_key]["curve"] = regress[2]
        return {field: retval}

    def kinetic_energy_spectra(self) -> dict[str, NDArray | float]:
        """
        Adapted from Federrath's KE Spectra Python code

        Returns
        -------
        dict[str, NDArray | float]
            _description_
        """

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

    def structure_functions(
        self,
        num_seps: int = 100,
        num_points: int = 10000,
        sep_bounds: list[float] = [0.0, 1.0],
        log_scale: bool = True,
        anistropic: bool = False,
    ) -> dict[str, dict[str, NDArray] | NDArray]:

        vels: list[str] = ["velx", "vely", "velz"][: self.ndim]

        if log_scale:
            separations: NDArray = np.geomspace(*sep_bounds, num_seps)
        else:
            separations: NDArray = np.linspace(*sep_bounds, num_seps)

        vsfs: dict[str, dict[str, NDArray] | NDArray] = {"transverse": {}, "longitudinal": {}}

        shape: tuple[int] = (num_seps, num_points, self.ndim, 2)
        dtype = np.float64()
        win: None | MPI.Win = mpi.reallocate(
            id="pt_coords",
            nbytes=dtype.itemsize * np.prod(shape),
            itemsize=dtype.itemsize,
        )
        buffer: MPI.buffer = win.Shared_query(0)[0]

        # Initialize the numpy array with float64 buffer
        pt_coords = np.ndarray(buffer=buffer, dtype=dtype, shape=shape)

        shape: tuple[int] = (num_seps, num_points, self.ndim)
        dtype = np.float64()
        win: None | MPI.Win = mpi.reallocate(
            id="vel_comps",
            nbytes=dtype.itemsize * np.prod(shape),
            itemsize=dtype.itemsize,
        )
        buffer: MPI.buffer = win.Shared_query(0)[0]

        # Initialize the numpy array with float64 buffer
        vel_comps = np.ndarray(buffer=buffer, dtype=dtype, shape=shape)

        for order in range(1, 11):
            mpi.comm.barrier()

            pt_coords[...] = 0.0
            vel_comps[...] = 0.0

            pt_rand_1: NDArray = np.empty((num_points, self.ndim))
            pt_rand_2: NDArray = np.empty((num_points, self.ndim))
            lb, ub = mpi.parallel_range(iterations=num_seps)
            
            for i in range(lb, ub):

                sep: float = separations[i]

                pt_rand_1[:, :] = (
                    np.random.random(pt_rand_1.shape) * np.diff(self.domain_bounds, axis=1).ravel()
                    + self.domain_bounds[:, 0].ravel()
                )

                phi: NDArray = 2.0 * np.pi * np.random.random(num_points)
                theta: NDArray = np.arccos(2.0 * np.random.random(num_points) - 1.0)

                pt_rand_2[:, 0] = pt_rand_1[:, 0] + sep * np.sin(theta) * np.cos(phi)
                pt_rand_2[:, 1] = pt_rand_1[:, 1] + sep * np.sin(theta) * np.sin(phi)
                pt_rand_2[:, 2] = pt_rand_1[:, 2] + sep * np.cos(theta)

                # X-direction
                while np.any(pt_rand_2[:, 0] > self.xmax):
                    pt_rand_2[pt_rand_2[:, 0] > self.xmax, 0] += self.xmin - self.xmax

                while np.any(pt_rand_2[:, 0] < self.xmin):
                    pt_rand_2[pt_rand_2[:, 0] < self.xmin, 0] += self.xmax - self.xmin

                # Y-direction
                while np.any(pt_rand_2[:, 1] > self.ymax):
                    pt_rand_2[pt_rand_2[:, 1] > self.ymax, 1] += self.ymin - self.ymax

                while np.any(pt_rand_2[:, 1] < self.ymin):
                    pt_rand_2[pt_rand_2[:, 1] < self.ymin, 1] += self.ymax - self.ymin

                # Z-direction
                while np.any(pt_rand_2[:, 2] > self.zmax):
                    pt_rand_2[pt_rand_2[:, 2] > self.zmax, 2] += self.zmin - self.zmax

                while np.any(pt_rand_2[:, 2] < self.zmin):
                    pt_rand_2[pt_rand_2[:, 2] < self.zmin, 2] += self.zmax - self.zmin

                # Get velocity components
                cell_size: NDArray = np.diff(self.domain_bounds, axis=1).flatten() / self.nCellsVec

                pt1: NDArray = np.empty(pt_rand_1.shape, dtype=int)
                pt2: NDArray = np.empty(pt_rand_2.shape, dtype=int)

                for j in range(self.ndim):
                    pt1[:, j] = np.floor((pt_rand_1[:, j] - self.domain_bounds[j, 0]) / cell_size[j]).astype(
                        int
                    )
                    pt2[:, j] = np.floor((pt_rand_2[:, j] - self.domain_bounds[j, 0]) / cell_size[j]).astype(
                        int
                    )

                for j, velocity in enumerate(vels):
                    vel_comps[i, :, j] = (
                        self.data(velocity)[pt2[:, 0], pt2[:, 1], pt2[:, 2]]
                        - self.data(velocity)[pt1[:, 0], pt1[:, 1], pt1[:, 2]]
                    )

                pt_coords[i, ..., 0] = pt_rand_1[...]
                pt_coords[i, ..., 1] = pt_rand_2[...]

            mpi.comm.barrier()
            sep_vec: NDArray = pt_coords[..., 1] - pt_coords[..., 0]

            rhat: NDArray = np.empty_like(sep_vec)

            if anistropic:
                rhat[..., 0] = 1.0
                rhat[..., 1:] = 0.0
            else:
                for i in range(self.ndim):
                    rhat[..., i] = sep_vec[..., i] / np.sqrt(np.sum(sep_vec**2, axis=2))

            long_comp: NDArray = np.abs(np.sum(vel_comps * rhat, axis=2))

            long_vsfs: NDArray = np.sum(long_comp**order, axis=1) / float(num_points)

            long_dvel: NDArray = np.empty_like(sep_vec)
            for i in range(self.ndim):
                long_dvel[..., i] = long_comp * rhat[..., i]

            trans_comp: NDArray = np.sqrt(np.sum((vel_comps - long_dvel) ** 2, axis=2))
            trans_vsfs: NDArray = np.sum(trans_comp**order, axis=1) / float(num_points)

            vsfs["transverse"][f"{order}"] = np.copy(trans_vsfs)
            vsfs["longitudinal"][f"{order}"] = np.copy(long_vsfs)
            vsfs["separations"] = separations

        mpi.deallocate(id="pt_coords")
        mpi.deallocate(id="vel_comps")
        return vsfs

    def mass_fraction(self, masks: dict[str, NDArray] | None = None) -> NDArray:

        mass: NDArray = self.data("dens") * self.cell_volume_min

        xfrac: dict[str, NDArray] = {"total": np.sum(mass)}
        for name, mask in masks.items():
            xfrac[name] = np.sum(mass[mask])

        del mass
        return xfrac
