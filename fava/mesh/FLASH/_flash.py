import logging
import itertools
import sys
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from functools import cached_property

import h5py
import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from mpi4py import MPI

from fava.geometry import AXIS, EDGE, GEOMETRY
from fava.mesh.structured import Structured
from fava.mesh.FLASH._util import FIELD_MAPPING, NGUARD, MESH_MDIM
from fava.model import Model
from fava.util import mpi, NP_T, HID_T

logger: logging.Logger = logging.getLogger(__file__)


class BLOCK_TYPE(Enum):
    LEAF = 1
    PARENT = 2
    ANCESTOR = 3
    IBDRY = 200
    JBDRY = 201
    KBDRY = 202
    ANY_BDRY = 203
    ACTIVE = 204
    ALL = 205
    TRAVERSED = 254
    REFINEMENT = 321
    TRAVERSED_AND_ACTIVE = 278


@Model.register_mesh()
class FLASH(Structured):

    _filename: Path | None = None
    _loaded: bool = False
    fields: Dict[str, str] = {}

    def __init__(self, filename: Optional[str | Path] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.filename: str | Path | None = filename
        self._chk_file: bool = False

    @classmethod
    def is_this_your_mesh(self, filename: str | Path, *args, **kwargs) -> bool:
        fn_types: Tuple[str, str] = ("hdf5_chk_", "hdf5_plt_cnt_")
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

        if "chk" in self._filename.stem:
            self._chk_file = True

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
            self._data: Dict = {}
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

                try:
                    self._mpi_assign_blocks()
                except Exception as exc:
                    logger.exception("Error assigning blocks with MPI", exc_info=True)
                    raise RuntimeError from exc

                self._read_nvar_names(handle=f)
                self._read_bflags(handle=f)
                self._read_coordinates(handle=f)
                self._read_block_sizes(handle=f)
                self._read_bounds(handle=f)
                self._read_processor_numbers(handle=f)
                self._read_node_type(handle=f)
                self._read_refine_level(handle=f)
                self._read_gid(handle=f)
                self._read_which_child(handle=f)

        except Exception as exc:
            logger.exception("Error reading FLASH FILE %s", self.filename, exc_info=True)
            raise RuntimeError from exc

    # MPI-heavy
    def _mpi_assign_blocks(self) -> None:
        if self.nblocks < 1:
            raise ValueError("[FLASH reader] _mpi_assign_blocks must come after the general read function.")

        extra: int = self.nblocks % mpi.procs

        self.nblocks_local: int = self.nblocks // mpi.procs
        if mpi.id < extra:
            self.nblocks_local += 1

    def _mpi_get_global_block_id(self, blkid: int) -> int:
        if self.nblocks_local < 1:
            raise ValueError(
                "[FLASH reader] _mpi_get_global_block_id must come after _mpi_assigned_blocks functions."
            )

        extra: int = self.nblocks % mpi.procs

        if mpi.id < extra:
            return mpi.id * self.nblocks_local + blkid
        else:
            return extra * (self.nblocks_local + 1) + (mpi.id - extra) * self.nblocks_local + blkid

    def _mpi_get_local_block_id(self, blkid: int) -> int:
        if self.nblocks_local < 1:
            raise ValueError(
                "[FLASH reader] _mpi_get_local_block_id must come after _mpi_assigned_blocks functions."
            )

        extra: int = self.nblocks % mpi.procs

        if mpi.id < extra:
            return blkid * self.nblocks_local
        else:
            return (blkid - (extra * (self.nblocks_local + 1))) % self.nblocks_local

    @cached_property
    def blk_beg(self) -> int:
        return self._mpi_get_global_block_id(0)

    @cached_property
    def blk_end(self) -> int:
        return self.blk_beg + self.nblocks_local

    # READERS
    def _read_scalars(self, handle: h5py.File) -> None:
        try:
            self.scalars: Dict[Dict[str, Any]] = {"real": {}, "integer": {}, "logical": {}, "string": {}}
            for key in self.scalars.keys():
                tmp: h5py.Dataset = handle[f"{key} scalars"]
                if key != "string":
                    self.scalars[key] = dict(zip(np.char.strip(tmp[:, "name"].astype(str)), tmp[:, "value"]))
                else:
                    self.scalars[key] = dict(
                        zip(
                            np.char.strip(tmp[:, "name"].astype(str)),
                            np.char.strip(tmp[:, "value"].astype(str)),
                        )
                    )
        except Exception as exc:
            logger.exception("Error occurred in reading FLASH SCALARS", exc_info=True)
            raise RuntimeError from exc

    def _read_runtime_parameters(self, handle: h5py.File) -> None:
        try:
            self.runtime_parameters: Dict[Dict[str, Any]] = {
                "real": {},
                "integer": {},
                "logical": {},
                "string": {},
            }
            for key in self.runtime_parameters.keys():
                tmp: h5py.Dataset = handle[f"{key} runtime parameters"]
                if key != "string":
                    self.runtime_parameters[key] = dict(
                        zip(np.char.strip(tmp[:, "name"].astype(str)), tmp[:, "value"])
                    )
                else:
                    self.runtime_parameters[key] = dict(
                        zip(
                            np.char.strip(tmp[:, "name"].astype(str)),
                            np.char.strip(tmp[:, "value"].astype(str)),
                        )
                    )
        except Exception as exc:
            logger.exception("Error occurred in reading FLASH RUNTIME PARAMETERS", exc_info=True)
            raise RuntimeError from exc

    def _read_nvar_names(self, handle: h5py.File) -> None:
        try:
            # np.char.
            self.fields: NDArray = np.squeeze(handle["unknown names"][()]).astype(str)

        except Exception as exc:
            logger.exception("Error occurred in reading FLASH UNK VARIABLE NAMES", exc_info=True)
            raise RuntimeError from exc

    def _read_simulation_parameters(self, handle: h5py.File) -> None:
        try:
            raise NotImplementedError("Reading Simulation Parameters not implemented!")
        except Exception as exc:
            logger.exception("Error occurred in reading FLASH Simulation Parameters", exc_info=True)
            raise RuntimeError from exc

    def _read_bflags(self, handle: h5py.File) -> None:
        key: str = "bflags"
        self.bflags = self._read_shared_array(handle=handle, key=key)

    def _read_coordinates(self, handle: h5py.File) -> None:
        key: str = "coordinates"
        self.coordinates = self._read_shared_array(handle=handle, key=key)

    def _read_block_sizes(self, handle: h5py.File) -> None:
        key: str = "block size"
        self.block_size = self._read_shared_array(handle=handle, key=key)

    def _read_bounds(self, handle: h5py.File) -> None:
        key: str = "bounding box"
        self.block_bounds = self._read_shared_array(handle=handle, key=key)

    def _read_processor_numbers(self, handle: h5py.File) -> None:
        key: str = "processor number"
        self.processors = self._read_shared_array(handle=handle, key=key, dtype=NP_T.INT64)

    def _read_node_type(self, handle: h5py.File) -> None:
        key: str = "node type"
        self.node_type = self._read_shared_array(handle=handle, key=key, dtype=NP_T.INT64)

    def _read_refine_level(self, handle: h5py.File) -> None:
        key: str = "refine level"
        self.refine_level = self._read_shared_array(handle=handle, key=key, dtype=NP_T.INT64)

    def _read_gid(self, handle: h5py.File) -> None:
        key: str = "gid"
        self.gid = self._read_shared_array(handle=handle, key=key, dtype=NP_T.INT64)

    def _read_which_child(self, handle: h5py.File) -> None:
        key: str = "which child"
        self.which_child = self._read_shared_array(handle=handle, key=key, dtype=NP_T.INT64)

    def _read_variable_data(self, handle: h5py.File, name: str) -> None:
        try:
            if name not in handle:
                raise KeyError(f"{name} field not found in dataset {self.filename}")

            dataset: h5py.Group | h5py.Dataset | h5py.Datatype = handle[name]
            shape: list[int] = list(dataset.shape)

            axis1: int = shape[-3]
            axis2: int = shape[-1]
            shape[-3] = axis2
            shape[-1] = axis1

            # We want to store the arrays in as 64-bit reals, even if the file holds 32-bit reals
            win: None | MPI.Win = mpi.reallocate(
                id=name, nbytes=MPI.DOUBLE.Get_size() * np.prod(shape), itemsize=MPI.DOUBLE.Get_size()
            )
            buffer: MPI.buffer = win.Shared_query(0)[0]

            # Initialize the numpy array with float64 buffer
            self._data[name] = np.ndarray(buffer=buffer, dtype=np.float64, shape=shape)

            mpi.comm.barrier()

            # We don't really need all processes reading in the data, let root process handle it
            if mpi.root:
                _temp_array: NDArray = np.ascontiguousarray(
                    np.swapaxes(dataset[()].astype(np.float64), axis1=-1, axis2=-3)
                )
                self._data[name][...] = _temp_array

            mpi.comm.barrier()

        except Exception as exc:
            logger.exception("Error occurred in reading FLASH VARIABLE DATA", exc_info=True)
            raise RuntimeError from exc

    def _read_shared_array(self, handle: h5py.File, key: str, dtype=None) -> NDArray:
        try:

            dataset: h5py.Group | h5py.Dataset | h5py.Datatype = handle[key]
            shape: tuple = dataset.shape
            dtype_: Any = dataset.dtype if dtype is None else dtype

            if dtype is None:
                win: None | MPI.Win = mpi.reallocate(id=key, nbytes=dataset.nbytes, itemsize=dtype_.itemsize)
                buffer = win.Shared_query(0)[0]
                array: NDArray = np.ndarray(buffer=buffer, dtype=dtype_, shape=shape)
                dataset.read_direct(array)

            else:
                tmp_ = dataset[()].astype(dtype)
                win: None | MPI.Win = mpi.reallocate(id=key, nbytes=tmp_.nbytes, itemsize=dtype_.itemsize)
                buffer = win.Shared_query(0)[0]
                array: NDArray = np.ndarray(buffer=buffer, dtype=dtype_, shape=shape)
                array[...] = tmp_[...]

            mpi.comm.barrier()
            return array
        except Exception as exc:
            logger.exception("Error occurred in reading FLASH DATAFILE: <%s>", key, exc_info=True)
            raise RuntimeError from exc

    # SETTERS
    def _set_reals(self) -> None:
        self.time: float = np.float64(self.scalars["real"].get("time"))
        self.xmin: float = np.float64(self.runtime_parameters["real"].get("xmin", 0))
        self.xmax: float = np.float64(self.runtime_parameters["real"].get("xmax", 1))
        self.ymin: float = np.float64(self.runtime_parameters["real"].get("ymin", 0))
        self.ymax: float = np.float64(self.runtime_parameters["real"].get("ymax", 1))
        self.zmin: float = np.float64(self.runtime_parameters["real"].get("zmin", 0))
        self.zmax: float = np.float64(self.runtime_parameters["real"].get("zmax", 1))

    def _set_integers(self) -> None:
        self.ndim: int = np.int64(self.scalars["integer"].get("dimensionality"))
        self.nxb: int = np.int64(self.scalars["integer"].get("nxb"))
        self.nyb: int = np.int64(self.scalars["integer"].get("nyb"))
        self.nzb: int = np.int64(self.scalars["integer"].get("nzb"))
        self.iprocs: int = np.int64(self.scalars["integer"].get("iprocs"))
        self.jprocs: int = np.int64(self.scalars["integer"].get("jprocs"))
        self.kprocs: int = np.int64(self.scalars["integer"].get("kprocs"))

        self.nblockx: int = np.int64(self.runtime_parameters["integer"].get("nblockx"))
        self.nblocky: int = np.int64(self.runtime_parameters["integer"].get("nblocky"))
        self.nblockz: int = np.int64(self.runtime_parameters["integer"].get("nblockz"))
        self.nblocks: int = np.int64(
            self.scalars["integer"].get("total blocks", self.scalars["integer"].get("globalnumblocks"))
        )

    @property
    def domain_bounds(self) -> NDArray:
        return np.array(
            [[self.xmin, self.xmax], [self.ymin, self.ymax], [self.zmin, self.zmax]], dtype=np.float64
        )

    @property
    def ncells(self) -> int:
        return self.nxb * self.nyb * self.nzb

    @property
    def nCellsVec(self) -> NDArray:
        return np.array([self.nxb, self.nyb, self.nzb], dtype=np.int32)

    @property
    def nBlksVec(self) -> NDArray:
        return np.array([self.nblockx, self.nblocky, self.nblockz], dtype=np.int32)

    @property
    def nblockx(self) -> int:
        return self._nblockx

    @nblockx.setter
    def nblockx(self, n: int) -> None:
        if "nblockx" in self.scalars["integer"]:
            self.scalars["integer"]["nblockx"] = n
        if "nblockx" in self.runtime_parameters["integer"]:
            self.runtime_parameters["integer"]["nblockx"] = n
        self._nblockx: int = n

    @property
    def nblocky(self) -> int:
        return self._nblocky

    @nblocky.setter
    def nblocky(self, n: int) -> None:
        if "nblocky" in self.scalars["integer"]:
            self.scalars["integer"]["nblocky"] = n
        if "nblocky" in self.runtime_parameters["integer"]:
            self.runtime_parameters["integer"]["nblocky"] = n
        self._nblocky: int = n

    @property
    def nblockz(self) -> int:
        return self._nblockz

    @nblockz.setter
    def nblockz(self, n: int) -> None:
        if "nblockz" in self.scalars["integer"]:
            self.scalars["integer"]["nblockz"] = n
        if "nblockz" in self.runtime_parameters["integer"]:
            self.runtime_parameters["integer"]["nblockz"] = n
        self._nblockz: int = n

    @property
    def nxb(self) -> int:
        return self._nxb

    @nxb.setter
    def nxb(self, n: int) -> None:
        if "nxb" in self.scalars["integer"]:
            self.scalars["integer"]["nxb"] = n
        if "nxb" in self.runtime_parameters["integer"]:
            self.runtime_parameters["integer"]["nxb"] = n
        self._nxb: int = n

    @property
    def nyb(self) -> int:
        return self._nyb

    @nyb.setter
    def nyb(self, n: int) -> None:
        if "nyb" in self.scalars["integer"]:
            self.scalars["integer"]["nyb"] = n
        if "nyb" in self.runtime_parameters["integer"]:
            self.runtime_parameters["integer"]["nyb"] = n
        self._nyb: int = n

    @property
    def nzb(self) -> int:
        return self._nzb

    @nzb.setter
    def nzb(self, n: int) -> None:
        if "nzb" in self.scalars["integer"]:
            self.scalars["integer"]["nzb"] = n
        if "nzb" in self.runtime_parameters["integer"]:
            self.runtime_parameters["integer"]["nzb"] = n
        self._nzb: int = n

    @property
    def nblocks(self) -> int:
        return self._nblocks

    @nblocks.setter
    def nblocks(self, n: int) -> None:
        if "globalnumblocks" in self.scalars["integer"]:
            self.scalars["integer"]["globalnumblocks"] = n
        if "globalnumblocks" in self.runtime_parameters["integer"]:
            self.runtime_parameters["integer"]["globalnumblocks"] = n
        self._nblocks: int = n

    @property
    def xmin(self) -> float:
        return self._xmin

    @xmin.setter
    def xmin(self, n: float) -> None:
        if "xmin" in self.scalars["real"]:
            self.scalars["real"]["xmin"] = n
        if "xmin" in self.runtime_parameters["real"]:
            self.runtime_parameters["real"]["xmin"] = n
        self._xmin: float = n

    @property
    def xmax(self) -> float:
        return self._xmax

    @xmax.setter
    def xmax(self, n: float) -> None:
        if "xmax" in self.scalars["real"]:
            self.scalars["real"]["xmax"] = n
        if "xmax" in self.runtime_parameters["real"]:
            self.runtime_parameters["real"]["xmax"] = n
        self._xmax: float = n

    @property
    def ymin(self) -> float:
        return self._ymin

    @ymin.setter
    def ymin(self, n: float) -> None:
        if "ymin" in self.scalars["real"]:
            self.scalars["real"]["ymin"] = n
        if "ymin" in self.runtime_parameters["real"]:
            self.runtime_parameters["real"]["ymin"] = n
        self._ymin: float = n

    @property
    def ymax(self) -> float:
        return self._ymax

    @ymax.setter
    def ymax(self, n: float) -> None:
        if "ymax" in self.scalars["real"]:
            self.scalars["real"]["ymax"] = n
        if "ymax" in self.runtime_parameters["real"]:
            self.runtime_parameters["real"]["ymax"] = n
        self._ymax: float = n

    @property
    def zmin(self) -> float:
        return self._zmin

    @zmin.setter
    def zmin(self, n: float) -> None:
        if "zmin" in self.scalars["real"]:
            self.scalars["real"]["zmin"] = n
        if "zmin" in self.runtime_parameters["real"]:
            self.runtime_parameters["real"]["zmin"] = n
        self._zmin: float = n

    @property
    def zmax(self) -> float:
        return self._zmax

    @zmax.setter
    def zmax(self, n: float) -> None:
        if "zmax" in self.scalars["real"]:
            self.scalars["real"]["zmax"] = n
        if "zmax" in self.runtime_parameters["real"]:
            self.runtime_parameters["real"]["zmax"] = n
        self._zmax: float = n

    # DELETERS
    def _delete_cached_properties(self) -> None:
        if "geometry" in self.__dict__:
            del self.geometry
        if "domain_volume" in self.__dict__:
            del self.domain_volume
        if "cell_volume_min" in self.__dict__:
            del self.cell_volume_min
        if "cell_volume_max" in self.__dict__:
            del self.cell_volume_max
        # if "cell_volumes" in self.__dict__:
        #     del self.cell_volumes
        if "refine_level_max" in self.__dict__:
            del self.refine_level_max

    # CACHED PROPERTIES
    @cached_property
    def geometry(self) -> GEOMETRY:
        return GEOMETRY(self.scalars["string"].get("geometry", "").lower())

    @cached_property
    def refine_level_max(self) -> int:
        return self.refine_level.max()

    @cached_property
    def domain_volume(self) -> float:
        match self.geometry:
            case GEOMETRY.CARTESIAN:
                vol: float = np.prod(np.diff(self.domain_bounds))

            case _:
                msg: str = f"Domain volume not implemented for {self.geometry}"
                logger.exception(msg)
                raise NotImplementedError(msg)

        return vol

    @cached_property
    def cell_volume_max(self) -> float:
        return self.get_cell_volume_from_refinement()

    @cached_property
    def cell_volume_min(self) -> float:
        return self.get_cell_volume_from_refinement(self.refine_level.max())

    def get_cell_volumes(self, block_type="LEAF") -> np.ndarray:
        blocklist: NDArray = self.get_blocklist(block_type=block_type)
        volumes: NDArray = np.zeros(blocklist.size, dtype=np.float64)
        for lb, blk in enumerate(blocklist):
            volumes[lb] = self.get_cell_volume_from_refinement(refine_level=self.refine_level[blk])
        return volumes

    def save(self, filename: str | Path | None = None, names: List[str] = None) -> None:

        old_filename: Path = self.filename
        if filename is not None:
            self.filename = filename

        try:
            with h5py.File(name=self.filename, mode="w") as f:
                self._write_file_type(handle=f)
                self._write_simulation_parameters(handle=f)
                self._write_sim_info(handle=f)
                self._write_parameters(handle=f)
                self._write_block_coordinates(handle=f)
                self._write_block_sizes(handle=f)
                self._write_block_bounds(handle=f)
                self._write_block_node_type(handle=f)
                self._write_block_refine_level(handle=f)
                self._write_block_gid(handle=f)
                self._write_which_child(handle=f)
                self._write_bflags(handle=f)

                # Expensive portion of the file saving, writing the variable data for UNK array
                names_: List[str] = names if names is not None else self._data.keys()
                self._write_nvars(handle=f, names=names_)
                self._write_variable_data(handle=f, names=names_)

        except Exception as exc:
            logger.exception("Error saving FLASH FILE %s", self.filename, exc_info=True)
            raise RuntimeError from exc
        finally:
            self.filename = old_filename

    def _write_file_type(self, handle: h5py.File) -> None: ...

    def _write_simulation_parameters(self, handle: h5py.File) -> None: ...

    def _write_sim_info(self, handle: h5py.File) -> None: ...

    def _write_parameters(self, handle: h5py.File) -> None:
        """
        Writes out both runtime parameters and scalars sets to FLASH file

        Parameters
        ----------
        handle : h5py.File
            File handle for an open HDF5 file to write to
        """
        for key in self.scalars.keys():

            # Grab the correct compound data type to save parameter set to FLASH file
            match key:
                case "real":
                    DTYPE = HID_T.F64_PARAMETER
                case "integer":
                    DTYPE = HID_T.I32_PARAMETER
                case "logical":
                    DTYPE = HID_T.BOOL_PARAMETER
                case "string":
                    # We need to prep the strings for saving to FLASH file
                    DTYPE = HID_T.STR_PARAMETER
                    data = [(f"{k:256s}", f"{v:256s}") for k, v in self.runtime_parameters[key].items()]
                    handle.create_dataset(
                        name=f"{key} runtime parameters",
                        shape=len(data),
                        dtype=DTYPE,
                        data=data,
                    )

                    [(f"{k:256s}", f"{v:256s}") for k, v in self.scalars[key].items()]
                    handle.create_dataset(
                        name=f"{key} scalars",
                        shape=len(data),
                        dtype=DTYPE,
                        data=data,
                    )
                    # String params saved, move to next key
                    continue
                case _:
                    logger.warning("Do not recognize parameter set %s", key)
                    DTYPE = None

            # Trying to save an unrecognized parameter set, skip it
            if DTYPE is None:
                continue

            # Real, integer, and logical keys get saved here
            data = [(f"{k:256s}", v) for k, v in self.runtime_parameters[key].items()]
            handle.create_dataset(
                name=f"{key} runtime parameters",
                shape=len(data),
                dtype=DTYPE,
                data=data,
            )

            data = [(f"{k:256s}", v) for k, v in self.scalars[key].items()]
            handle.create_dataset(
                name=f"{key} scalars",
                shape=len(data),
                dtype=DTYPE,
                data=data,
            )

    def _write_block_coordinates(self, handle: h5py.File) -> None:
        DTYPE = HID_T.F64 if self._chk_file else HID_T.F32
        handle.create_dataset(
            name="coordinates", shape=self.coordinates.shape, dtype=DTYPE, data=self.coordinates
        )

    def _write_block_sizes(self, handle: h5py.File) -> None:
        # PLOT FILE GET F32, CHECKPOINT FILE GET F64; add this in later
        DTYPE = HID_T.F64 if self._chk_file else HID_T.F32
        handle.create_dataset(name="block size", shape=self.block_size.shape, dtype=DTYPE, data=self.block_size)

    def _write_block_bounds(self, handle: h5py.File) -> None:
        # PLOT FILE GET F32, CHECKPOINT FILE GET F64; add this in later
        DTYPE = HID_T.F64 if self._chk_file else HID_T.F32
        handle.create_dataset(
            name="bounding box", shape=self.block_bounds.shape, dtype=DTYPE, data=self.block_bounds
        )

    def _write_block_node_type(self, handle: h5py.File) -> None:
        handle.create_dataset(
            name="node type", shape=self.node_type.shape, dtype=HID_T.I32, data=self.node_type.astype(np.int32)
        )

    def _write_block_refine_level(self, handle: h5py.File) -> None:
        handle.create_dataset(
            name="refine level",
            shape=self.refine_level.shape,
            dtype=HID_T.I32,
            data=self.refine_level.astype(np.int32),
        )

    def _write_block_gid(self, handle: h5py.File) -> None:
        handle.create_dataset(name="gid", shape=self.gid.shape, dtype=HID_T.I32, data=self.gid.astype(np.int32))

    def _write_processor_number(self, handle: h5py.File) -> None:
        handle.create_dataset(
            name="processor number",
            shape=self.processors.shape,
            dtype=HID_T.I32,
            data=self.processors.astype(np.int32),
        )

    def _write_bflags(self, handle: h5py.File) -> None:
        handle.create_dataset(name="bflags", shape=self.bflags.shape, dtype=HID_T.I32, data=self.bflags)

    def _write_which_child(self, handle: h5py.File) -> None:
        handle.create_dataset(
            name="which child",
            shape=self.which_child.shape,
            dtype=HID_T.I32,
            data=self.which_child.astype(np.int32),
        )

    def _write_nvars(self, handle: h5py.File, names: list[str] | None = None) -> None:
        fields = np.bytes_(names) if names is not None else self.fields.astype(HID_T.UNKNOWN_NAMES)
        handle.create_dataset(name="unknown names", shape=fields.shape, dtype=HID_T.UNKNOWN_NAMES, data=fields)

    def _write_variable_data(self, handle: h5py.File, names: List[str] = None) -> None:
        # Ensure we are using the correct HID_T float type here depending on FILE TYPE
        DTYPE = HID_T.F64 if self._chk_file else HID_T.F32
        for var in names:
            if var not in self._data:
                continue

            # Add existing data to FLASH file, this will ensure that we correctly swap the
            # GRID I and K axes whether we have a block-based or uniform grid (4-d vs 3-d)
            IAXIS: int = -3
            KAXIS: int = -1
            shape = list(self._data[var].shape)
            sx: int = shape[IAXIS]
            sz: int = shape[KAXIS]
            shape[IAXIS] = sz
            shape[KAXIS] = sx
            handle.create_dataset(
                name=var,
                shape=shape,
                dtype=DTYPE,
                data=np.swapaxes(self._data[var], axis1=IAXIS, axis2=KAXIS),
            )

    # GETTER METHODS

    def get_blocklist(self, block_type: str = "LEAF") -> NDArray:

        if isinstance(block_type, BLOCK_TYPE):
            btype: BLOCK_TYPE = block_type
        else:
            btype: BLOCK_TYPE = BLOCK_TYPE[block_type]

        lb: int = self.blk_beg
        ub: int = self.blk_end

        match btype:
            case BLOCK_TYPE.LEAF:
                return (lb + np.argwhere(self.node_type[lb:ub] == BLOCK_TYPE.LEAF.value).flatten()).astype(
                    np.int64
                )
            case BLOCK_TYPE.ALL:
                return np.arange(lb, ub, dtype=np.int64)
            case _:
                logger.error("Do not recognize BLOCK TYPE %s", btype.name)
                raise ValueError

    def get_cell_coords(
        self, axis: int, blockID: int = 0, edge: str = "CENTER", guardcell: bool = False
    ) -> NDArray:

        n = [self.nxb, self.nyb, self.nzb][axis]
        lb, ub = self.block_bounds[blockID, axis, :]
        dx = (ub - lb) / (float(n) + 1)

        m = n

        if guardcell:
            lb = lb - NGUARD * dx
            m += NGUARD

        match EDGE[edge]:
            case EDGE.CENTER:
                lb += 0.5 * dx
                ub -= 0.5 * dx
            case EDGE.RIGHT:
                lb += dx
            case EDGE.LEFT:
                ub -= dx

        return np.linspace(lb, ub, m)

    def get_point_data(self, blockID: int, point: List[int], field: str) -> float:

        match self.ndim:
            case 1:
                val: NDArray = self.data(field)[blockID, point[0]]
            case 2:
                val: NDArray = self.data(field)[blockID, point[0], point[1]]
            case 3:
                val: NDArray = self.data(field)[blockID, point[0], point[1], point[2]]
        return val

    def get_coord_index(self, point, block_list: List[int]) -> Tuple[List, int]:
        idx: List[None] = [None, None, None][: self.ndim]
        for blk in block_list:
            in_blk: bool = self.is_point_in_block(point=point, blockID=blk)

            if in_blk:
                xcoord: NDArray = self.get_cell_coords(axis=0, blockID=blk)
                idx[0] = (np.abs(xcoord - point[0])).argmin()

                if self.ndim > 1:
                    ycoord: NDArray = self.get_cell_coords(axis=1, blockID=blk)
                    idx[1] = (np.abs(ycoord - point[1])).argmin()

                if self.ndim > 2:
                    zcoord: NDArray = self.get_cell_coords(axis=2, blockID=blk)
                    idx[2] = (np.abs(zcoord - point[2])).argmin()

                break

        return idx, blk

    def points_within_block(
        self,
        points: List[float] | np.array,
        axis: int,
        blockID: int,
        return_indices=False,
    ) -> tuple[NDArray, NDArray]:
        box: NDArray = self.block_bounds[blockID, axis, :]
        if isinstance(points, list):
            points_: NDArray = np.array(points)
        else:
            points_: NDArray = np.copy(points)

        conditional: NDArray = (points_ >= box[0]) & (points_ <= box[1])
        if return_indices:
            return points_[conditional], np.argwhere(conditional).flatten()
        else:
            return points_[conditional]

    def is_point_in_block(self, point: List[float] | np.array, blockID: int) -> bool:

        box: NDArray = self.block_bounds[blockID, ...]

        is_in_box: bool = box[0, 0] <= point[0] < box[0, 1]
        if self.ndim > 1:
            is_in_box = is_in_box and (box[1, 0] <= point[1] < box[1, 1])
        if self.ndim > 2:
            is_in_box = is_in_box and (box[2, 0] <= point[2] < box[2, 1])

        return is_in_box

    def get_minimum_deltas(self, axis: int) -> Any:
        return (self.domain_bounds[axis, 1] - self.domain_bounds[axis, 0]) / (
            self.nCellsVec[axis] * self.nBlksVec[axis] * 2 ** (self.refine_level_max - 1)
        )

    def get_maximum_deltas(self, axis: int) -> Any:
        return (self.domain_bounds[axis, 1] - self.domain_bounds[axis, 0]) / (
            self.nCellsVec[axis] * self.nBlksVec[axis] * 2 ** (self.refine_level.min() - 1)
        )

    def get_deltas_from_refine_level(self, refine_level: int) -> List[float]:
        _res: list = []
        for i in range(self.ndim):
            _res.append(self.get_delta_from_refine_level(axis=i, refine_level=refine_level))
        return _res

    def get_delta_from_refine_level(self, axis: int, refine_level: int) -> float:
        return (self.domain_bounds[axis, 1] - self.domain_bounds[axis, 0]) / (
            self.nCellsVec[axis] * self.nBlksVec[axis] * 2 ** (refine_level - 1)
        )

    def get_block_deltas(self, blockID: int) -> List[float]:
        _res: list = []
        for i in range(self.ndim):
            _res.append(self.get_block_delta(axis=i, blockID=blockID))
        return _res

    def get_block_delta(self, axis: int, blockID: int) -> float:
        return (self.block_bounds[blockID, axis, 1] - self.block_bounds[blockID, axis, 0]) / (
            self.nCellsVec[axis]
        )

    def get_cell_volume_from_refinement(self, refine_level: int = 1) -> float:
        if self.geometry == GEOMETRY.CARTESIAN:
            cells = self.nxb * self.nblockx * 2 ** (refine_level - 1)
            if self.ndim > 1:
                cells *= self.nyb * self.nblocky * 2 ** (refine_level - 1)
            if self.ndim > 2:
                cells *= self.nzb * self.nblockz * 2 ** (refine_level - 1)
        return self.domain_volume / float(cells)

    def from_amr(
        self,
        subdomain_coords: NDArray | None = None,
        refine_level: int = -1,
        fields: list[str] | None = None,
        filename: Path | None = None,
    ) -> None:

        MPI_REAL: MPI.Datatype = MPI.DOUBLE if self._chk_file else MPI.FLOAT

        subdomain_flag: bool = any(0 not in sdc for sdc in subdomain_coords)

        if subdomain_flag:
            if subdomain_coords[0, 0] < self.xmin or self.xmax < subdomain_coords[0, 1]:
                return

            if self.ndim > 1:
                if subdomain_coords[1, 0] < self.ymin or self.ymax < subdomain_coords[1, 1]:
                    return

            if self.ndim > 2:
                if subdomain_coords[2, 0] < self.zmin or self.zmax < subdomain_coords[2, 1]:
                    return

        if mpi.root:
            logger.info("Starting Refinement")
            if subdomain_flag:
                logger.info("Targeting subdomain")

        # ref_lev_max: int = self.refine_level_max
        ref_lev_max: int = mpi.comm.allreduce(self.refine_level_max, MPI.MAX)

        if mpi.root:
            if refine_level > self.refine_level_max:
                logger.warning(
                    "Specified refinement level %d too, large, restricting mesh to max refinement level %d",
                    refine_level,
                    self.refine_level_max,
                )

        ref_lev: int = min(refine_level, ref_lev_max)

        if ref_lev > 0:
            ref_lev_max: int = ref_lev

        grid_bound_box: NDArray = np.zeros_like(self.block_bounds[0, ...])
        grid_bound_box[:, 0] = np.min(self.block_bounds[..., 0], axis=0)
        grid_bound_box[:, 1] = np.max(self.block_bounds[..., 1], axis=0)

        cellfac: int = 2 ** (ref_lev_max - 1)
        grid_delta: NDArray = (
            np.diff(grid_bound_box, axis=1).flatten() / (self.nCellsVec * self.nBlksVec * cellfac)
        )[:, None]
        grid_half_delta = grid_delta * 0.5

        local_BCIDs: NDArray = np.zeros((self.nblocks, MESH_MDIM, 2), dtype=np.int32)
        subdomain_BCIDs: NDArray = np.zeros((MESH_MDIM, 2), dtype=np.int32)

        for lb in range(self.nblocks):
            blk_bound_box: NDArray = self.block_bounds[lb]
            local_BCIDs[lb, :, :] = (blk_bound_box - grid_bound_box[:, 0, None] + grid_half_delta) / grid_delta

        if subdomain_flag:

            subdomain_BCIDs[:MESH_MDIM, :] = (
                0.5
                + (subdomain_coords[:MESH_MDIM, :] - grid_bound_box[:MESH_MDIM, :1]) / grid_delta[:MESH_MDIM, :]
            )

        max_scale: int = int(2 ** (ref_lev_max - 1))

        fine_blks = max_scale * np.array([self.nblockx, self.nblocky, self.nblockz], dtype=np.int32)
        fine_cells: NDArray = np.ones_like(fine_blks)
        subd_cells: NDArray = np.ones_like(fine_cells)
        valid_cells: NDArray = np.ones_like(subd_cells)
        remain_cells: NDArray = np.zeros_like(valid_cells)
        procs: NDArray = np.zeros_like(remain_cells)

        if subdomain_flag:
            subd_cells[: self.ndim] = np.diff(subdomain_BCIDs[: self.ndim, :]).flatten()

        if self.ndim == 1:
            procs[0] = mpi.procs

            if subdomain_flag:
                fine_cells[0] = subd_cells[0] // procs[0]
                remain_cells[0] = subd_cells[0] - fine_cells[0] * procs[0]

            else:
                fine_cells[0] = fine_blks[0] * self.nCellsVec[0] // procs[0]
                remain_cells[0] = fine_blks[0] * self.nCellsVec[0] - fine_cells[0] * procs[0]

            if subdomain_flag:
                if fine_cells[0]:
                    raise RuntimeError("BAD!")

            fine_cells[0] += np.ceil(1 * remain_cells[0] / procs[0])

            valid_cells[0] = fine_cells[0]
            if remain_cells[0] % procs[0] != 0:
                if mpi.id % procs[0] > remain_cells[0] % procs[0]:
                    valid_cells[0] -= 1

        elif self.ndim == 2:
            primefacs, nfacts = self._find_factors(mpi.procs)
            procs[: self.ndim] = 1

            for i in range(nfacts):
                if fine_blks[0] * self.nCellsVec[0] / procs[0] >= fine_blks[1] * self.nCellsVec[1] / procs[1]:
                    procs[0] *= primefacs[i]
                else:
                    procs[1] *= primefacs[i]

            if subdomain_flag:
                fine_cells[: self.ndim] = subd_cells[: self.ndim] / procs[: self.ndim]
                remain_cells[: self.ndim] = (
                    subd_cells[: self.ndim] - fine_cells[: self.ndim] * procs[: self.ndim]
                )
            else:
                fine_cells[: self.ndim] = (
                    fine_blks[: self.ndim] * self.nCellsVec[: self.ndim] / procs[: self.ndim]
                )
                remain_cells[: self.ndim] = (
                    fine_blks[: self.ndim] * self.nCellsVec[: self.ndim]
                    - fine_cells[: self.ndim] * procs[: self.ndim]
                )

            if fine_cells[0] <= 0 or fine_cells[1] <= 0:
                raise RuntimeError("BAD!")

            fine_cells[: self.ndim] += np.ceil(1 * remain_cells[: self.ndim] // procs[: self.ndim])

            valid_cells[0] = fine_cells[0]
            if remain_cells[0] % procs[0] != 0:
                if mpi.id % procs[0] > remain_cells[0] % procs[0]:
                    valid_cells[0] -= 1

            valid_cells[1] = fine_cells[1]
            if remain_cells[1] % procs[1] != 0:
                if mpi.id - (mpi.id % procs[0]) / procs[0] > remain_cells[1] % procs[1]:
                    valid_cells[1] -= 1

        elif self.ndim == 3:
            primefacs, nfacts = self._find_factors(procs=mpi.procs)
            procs[: self.ndim] = 1

            if mpi.root:
                print("Setting refinement cells - 3D", flush=True)
            for i in range(nfacts):
                if (
                    fine_blks[0] * self.nCellsVec[0] / procs[0] >= fine_blks[1] * self.nCellsVec[1] / procs[1]
                    and fine_blks[0] * self.nCellsVec[0] / procs[0]
                    >= fine_blks[2] * self.nCellsVec[2] / procs[2]
                ):
                    procs[0] *= primefacs[i]
                elif (
                    fine_blks[1] * self.nCellsVec[1] / procs[1] > fine_blks[0] * self.nCellsVec[0] / procs[0]
                    and fine_blks[1] * self.nCellsVec[1] / procs[1]
                    >= fine_blks[2] * self.nCellsVec[2] / procs[2]
                ):
                    procs[1] *= primefacs[i]
                else:
                    procs[2] *= primefacs[i]

            if subdomain_flag:
                fine_cells[: self.ndim] = subd_cells[: self.ndim] // procs[: self.ndim]
                remain_cells[: self.ndim] = (
                    subd_cells[: self.ndim] - fine_cells[: self.ndim] * procs[: self.ndim]
                )
            else:
                fine_cells[: self.ndim] = (
                    fine_blks[: self.ndim] * self.nCellsVec[: self.ndim] / procs[: self.ndim]
                )
                remain_cells[: self.ndim] = (
                    fine_blks[: self.ndim] * self.nCellsVec[: self.ndim]
                    - fine_cells[: self.ndim] * procs[: self.ndim]
                )

            if fine_cells[0] <= 0 or fine_cells[1] <= 0 or fine_cells[2] <= 0:
                raise RuntimeError("BAD!")

            fine_cells[: self.ndim] += np.ceil(1 * remain_cells[: self.ndim] // procs[: self.ndim])

            valid_cells[0] = fine_cells[0]
            if remain_cells[0] % procs[0] != 0:
                if mpi.id % procs[0] > remain_cells[0] % procs[0]:
                    valid_cells[0] -= 1

            valid_cells[1] = fine_cells[1]
            if remain_cells[1] % procs[1] != 0:
                if mpi.id - (mpi.id % procs[0]) / procs[0] > remain_cells[1] % procs[1]:
                    valid_cells[1] -= 1

            valid_cells[2] = fine_cells[2]
            if remain_cells[2] % procs[2] != 0:
                if (
                    mpi.id - (mpi.id % (procs[0] * procs[1])) / (procs[0] * procs[1])
                    > remain_cells[2] % procs[2]
                ):
                    valid_cells[1] -= 1

        # Left off at line 615 in Lavaflow Flash Uniform refinement .cxx
        leaves: int = 0
        leaf_IDs: List[int] = []
        if ref_lev > -1:
            local_BCIDs[:, self.ndim : MESH_MDIM, 1] = 0

            for lb in self.get_blocklist("ALL"):

                maybe_leaf: bool = (
                    self.node_type[lb] == 1 and self.refine_level[lb] < ref_lev
                ) or self.refine_level[lb] == ref_lev

                if maybe_leaf and self._intersects_subdomain(
                    local_BCIDs=local_BCIDs[lb, ...],
                    subdomain_BCIDs=subdomain_BCIDs,
                    subdomain_flag=subdomain_flag,
                ):
                    leaf_IDs.append(lb)
                    leaves += 1

        else:
            local_BCIDs[:, self.ndim : MESH_MDIM, 1] = 0
            for lb in self.get_blocklist("ALL"):
                if self.node_type[lb] == 1 and self._intersects_subdomain(
                    local_BCIDs=local_BCIDs[lb, ...],
                    subdomain_BCIDs=subdomain_BCIDs,
                    subdomain_flag=subdomain_flag,
                ):
                    leaf_IDs.append(lb)
                    leaves += 1

        if subdomain_flag:
            refdom_bound_box = grid_bound_box[:, :1] + subdomain_BCIDs * grid_delta

        else:
            refdom_bound_box: NDArray = np.copy(grid_bound_box)

        if subdomain_flag:
            total_cells: NDArray = np.copy(subd_cells)
        else:
            total_cells = np.ones_like(fine_cells)
            total_cells[: self.ndim] = fine_blks[: self.ndim] * self.nCellsVec[: self.ndim]

        _leaf_IDs: List[Any] = mpi.comm.allgather(leaf_IDs)
        leaf_IDs = []
        for _leaf_list in _leaf_IDs:
            leaf_IDs += _leaf_list

        leaves = len(leaf_IDs)

        if mpi.root:
            print(f"{total_cells=}", flush=True)

        mpi.comm.barrier()

        # Break up the work among the processors, this only works due to us sharing memory (a departure from typical MPI behavior)
        # ToDo: Implement a backup version for systems that don't support shared memory MPI (issue for older versions of MPI)
        cell_tuples: int = np.prod(self.nCellsVec)
        extra: int = cell_tuples % mpi.procs
        cells_local: int = cell_tuples // mpi.procs

        if mpi.id < extra:
            cells_begin: int = mpi.id * cells_local
        else:
            cells_begin: int = extra * (cells_local + 1) + (mpi.id - extra) * cells_local

        cells_end: int = cells_begin + cells_local

        # Construct a list of i,j,k combinations and divide them up based on the above load balancing results
        block_indices: NDArray = np.array(
            list(itertools.product(range(self.nxb), range(self.nyb), range(self.nzb))), dtype=np.int64
        )[cells_begin:cells_end, :]

        # We need a shared memory location for the temporary variable during refinement, use shape (1, icells, jcells, kcells)
        shape = total_cells
        win: None | MPI.Win = mpi.reallocate(
            id="in_data", nbytes=np.prod(shape) * MPI.DOUBLE.Get_size(), itemsize=MPI.DOUBLE.Get_size()
        )
        buffer: MPI.buffer = win.Shared_query(0)[0]
        in_data = np.ndarray(buffer=buffer, shape=shape, dtype=np.float64)

        ttotal: float = 0.0

        mapping: dict = {}
        first: bool = True

        _fields: list[str] = fields if fields is not None else self.fields

        mpi.comm.barrier()
        for key in _fields:

            ti: float = time.time()

            in_data[...] = 0.0

            mpi.comm.barrier()

            if first:
                for leaf in leaf_IDs:

                    offx: int = local_BCIDs[leaf, 0, 0]
                    offy: int = local_BCIDs[leaf, 1, 0] if self.ndim > 1 else 0
                    offz: int = local_BCIDs[leaf, 2, 0] if self.ndim > 2 else 0
                    off_blk: NDArray = np.array([offx, offy, offz])

                    level_diff: int = ref_lev_max - self.refine_level[leaf]
                    scale: int = int(2**level_diff)

                    for i, j, k in block_indices:

                        ii_range: int = range(i * scale, (i + 1) * scale)
                        jj_range: int = range(
                            j * scale if self.ndim > 1 else 0, (j + 1) * scale if self.ndim > 1 else 1
                        )
                        kk_range: int = range(
                            k * scale if self.ndim > 2 else 0, (k + 1) * scale if self.ndim > 2 else 1
                        )

                        for ii, jj, kk in itertools.product(ii_range, jj_range, kk_range):
                            in_subdom_flag: bool = False

                            if subdomain_flag:
                                in_subdom_flag = self._inSubdomain(
                                    i=offx + ii,
                                    j=offy + jj,
                                    k=offz + kk,
                                    subdomain_BCIDs=subdomain_BCIDs,
                                )

                            indices: NDArray = np.array([ii, jj, kk], dtype=np.int32)

                            ind = off_blk + indices

                            map_data: bool = True
                            if subdomain_flag:
                                map_data = in_subdom_flag
                                if in_subdom_flag:
                                    ind -= subdomain_BCIDs[:, 0]

                            if map_data:
                                mapping[tuple(ind)] = (leaf, i, j, k)

            if mpi.root:
                print(f"Refining variable: {key}", flush=True)
                logger.info("Refining variable: %s", key)

            self.data(key)

            for dest, src in mapping.items():
                in_data[*dest] = self._data[key][*src]

            mpi.comm.barrier()
            win = mpi.reallocate(id=key, nbytes=in_data.nbytes, itemsize=in_data.itemsize)
            self._data[key] = np.ndarray(
                buffer=win.Shared_query(rank=0)[0], shape=in_data.shape, dtype=in_data.dtype
            )
            self._data[key][...] = in_data[...]

            mpi.comm.barrier()
            if mpi.root:
                if key == "dens":
                    print(
                        f"dens={self._data[key].sum()}, min={self._data[key].min()}, max={self._data[key].max()}",
                        flush=True,
                    )
                tf: float = time.time() - ti
                ttotal += tf
                print(f"\t/timer/ - {tf} sec", flush=True)

            first = False
            mpi.comm.barrier()

        if mpi.root:
            print(f"Total refinement time: {ttotal} sec", flush=True)

        self.gid: NDArray = -1 * np.ones(int(2 * self.ndim + 1 + 2**self.ndim), dtype=np.int32)
        self.refine_level: NDArray = np.ones(1, dtype=np.int32)
        self.node_type: NDArray = np.ones_like(self.refine_level)
        self.bflags: NDArray = -1 * np.ones_like(self.refine_level)
        self.which_child: NDArray = np.copy(self.which_child)
        self.nblockx = 1
        self.nblocky = 1
        self.nblockz = 1
        self.nblocks = 1
        self.nxb = total_cells[0]
        self.nyb = total_cells[1]
        self.nzb = total_cells[2]
        self.block_size: NDArray = (total_cells * grid_delta)[None, ...]
        self.block_bounds: NDArray = refdom_bound_box[None, ...]
        self.coordinates: NDArray = (0.5 * np.sum(refdom_bound_box, axis=1))[None, ...]

        self.xmin = refdom_bound_box[0, 0]
        self.xmax = refdom_bound_box[0, 1]
        self.ymin = refdom_bound_box[1, 0]
        self.ymax = refdom_bound_box[1, 1]
        self.zmin = refdom_bound_box[2, 0]
        self.zmax = refdom_bound_box[2, 1]

        mpi.comm.barrier()
        if mpi.root:
            print(f"Number of Data Keys {len(self._data.keys())}", flush=True)
            if filename is None:
                uni_stem: str = self.filename.stem
                uni_stem = uni_stem.replace("plt_cnt", "uniform")
                uni_stem = uni_stem.replace("chk", "uniform")
                uni_filename: Path = self.filename.with_stem(uni_stem)
            else:
                uni_filename = filename
            self.save(filename=uni_filename, names=_fields)

        # Deallocate the in_data shared array
        mpi.deallocate(id="in_data")
        mpi.comm.barrier()

    def _inSubdomain(self, i: int, j: int, k: int, subdomain_BCIDs) -> bool:
        return (
            subdomain_BCIDs[0, 0] <= i < subdomain_BCIDs[0, 1]
            and subdomain_BCIDs[1, 0] <= j < subdomain_BCIDs[1, 1]
            and subdomain_BCIDs[2, 0] <= k < subdomain_BCIDs[2, 1]
        )

    def _intersects_subdomain(self, local_BCIDs, subdomain_BCIDs, subdomain_flag: bool) -> bool:
        if not subdomain_flag:
            return True

        return all(
            subdomain_BCIDs[n, 0] <= local_BCIDs[n, 1] and local_BCIDs[n, 0] <= subdomain_BCIDs[n, 1]
            for n in range(MESH_MDIM)
        )

    def _find_factors(self, procs: int) -> Tuple[List[int], int]:
        prime_factors: List[int] = [0] * procs
        prime_factors[0] = procs

        factoring: int = 0
        factor_count: int = 1
        current_factor: int = -1
        i: int = 2

        factorized: bool = procs == 1
        factor_found: bool = False

        while not factorized:
            current_factor = prime_factors[factoring]
            while i < current_factor and not factor_found:
                if current_factor % i == 0:
                    prime_factors[factoring] = current_factor // i

                    prime_factors[factor_count] = i
                    factor_count += 1
                    factor_found = True
                i += 1

            i = 2
            factor_found = False
            if prime_factors[factoring] == current_factor:
                factoring += 1

            factorized = prime_factors[factoring] == 0

        return sorted(prime_factors, reverse=True), factoring

    def slice_average(self, field: str, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
        # field_: str | None = self.fields.get(field)
        # if field_ is None:
        field_: str = field

        ax_ = AXIS(axis)

        min_deltas: NDArray = np.array(
            [self.get_minimum_deltas(ii) for ii in range(self.ndim)], dtype=np.float64
        )

        match ax_:
            case AXIS.I:
                layer_volume: float = (self.ymax - self.ymin) * (self.zmax - self.zmin)
            case AXIS.J:
                layer_volume: float = (self.xmax - self.xmin) * (self.zmax - self.zmin)
            case AXIS.K:
                layer_volume: float = (self.ymax - self.ymin) * (self.xmax - self.xmin)
            case _:
                raise ValueError(f"Do not recognize AXIS enumeration {ax_}")

        span, alp = self.slice_integral(field_, axis=ax_)
        return span, alp / (min_deltas[ax_.value] * layer_volume)

    def slice_integral(self, field: str, axis: int = 0):
        # field_: str | None = self._fields.get(field)
        # if field_ is None:
        field_: str = field

        lrefcells: int = 2 ** (self.refine_level_max - 1)
        dims: list = [
            nb * bl * lrefcells for nb, bl in zip(self.nCellsVec[: self.ndim], self.nBlksVec[: self.ndim])
        ]

        ax_ = AXIS(axis)

        min_deltas: NDArray = np.array(
            [self.get_minimum_deltas(ii) for ii in range(self.ndim)], dtype=np.float64
        )

        match ax_:
            case AXIS.I:
                rmin, rmax = self.xmin, self.xmax
                nrb: int = self.nxb
            case AXIS.J:
                rmin, rmax = self.ymin, self.ymax
                nrb = self.nyb
            case AXIS.K:
                rmin, rmax = self.zmin, self.zmax
                nrb = self.nzb
            case _:
                raise ValueError(f"Do not recognize AXIS enumeration {ax_}")

        span: NDArray = np.linspace(rmin, rmax, dims[ax_.value] + 1, dtype=np.float64)

        blocklist: NDArray = self.get_blocklist()
        alp: NDArray = np.zeros(dims[ax_.value], dtype=np.float64)
        vol_fracs: NDArray = self.get_cell_volumes() * (
            min_deltas[ax_.value]
            / self.get_delta_from_refine_level(axis=ax_.value, refine_level=self.refine_level[blocklist])
        )

        for lb, blk in enumerate(blocklist):
            lref_n: int = int(2 ** (self.refine_level_max - 1) / 2 ** (self.refine_level[blk] - 1))
            lo: float = self.block_bounds[blk, axis, 0]
            ilo: int = np.argmin(np.abs(span[:-1] - lo))

            mean: NDArray = np.einsum("ijk->i", self.data(field_)[blk, ...]) * vol_fracs[lb]
            for i in range(nrb):
                jlo: int = ilo + i * lref_n
                jhi: int = ilo + (i + 1) * lref_n
                alp[jlo:jhi] += mean[i]

        global_alp: NDArray = np.zeros_like(alp)

        mpi.comm.Allreduce(alp, global_alp)

        return span, global_alp

    def reynolds_stress(self, raxis: int = 0):
        lrefcells: int = 2 ** (self.refine_level_max - 1)
        dims: list[int] = [
            nb * bl * lrefcells for nb, bl in zip(self.nCellsVec[: self.ndim], self.nBlksVec[: self.ndim])
        ]

        ax_ = AXIS(raxis)

        min_deltas: NDArray = np.array(
            [self.get_minimum_deltas(ii) for ii in range(self.ndim)], dtype=np.float64
        )

        axes: str = "xyz"[: self.ndim]

        nrb: int = 0
        layer_volume: float = 0.0
        rmin: float = 0.0
        rmax: float = 0.0

        match ax_:
            case AXIS.I:
                layer_volume = (self.ymax - self.ymin) * (self.zmax - self.zmin)
                rmin, rmax = self.xmin, self.xmax
                nrb = self.nxb
            case AXIS.J:
                layer_volume = (self.xmax - self.xmin) * (self.zmax - self.zmin)
                rmin, rmax = self.ymin, self.ymax
                nrb = self.nyb
            case AXIS.K:
                layer_volume = (self.ymax - self.ymin) * (self.xmax - self.xmin)
                rmin, rmax = self.zmin, self.zmax
                nrb = self.nzb
            case _:
                raise ValueError(f"Do not recognize AXIS enumeration {ax_}")

        layer_volume *= min_deltas[ax_.value]

        # Radial nodes (not cell centers, but edges) for finest allowed resolution for raxis
        radius: NDArray = np.linspace(rmin, rmax, dims[ax_.value] + 1)

        stress: dict[str, NDArray] = {}
        means: dict[str, NDArray] = {"dens": np.zeros(dims[ax_.value])}

        for i in range(self.ndim):
            means[f"vel{axes[i]}"] = np.zeros(dims[ax_.value])

            for j in range(i, self.ndim):
                stress[f"R{axes[i]}{axes[j]}"] = np.zeros(dims[ax_.value])

        blocklist: NDArray = self.get_blocklist()
        mapping: NDArray = np.zeros((blocklist.size, nrb, 2), dtype=np.int64)

        vol_fracs: NDArray = self.get_cell_volumes() * (
            min_deltas[ax_.value]
            / self.get_delta_from_refine_level(axis=ax_.value, refine_level=self.refine_level[blocklist])
        )

        for lb, blk in enumerate(blocklist):
            lref_n: int = int(2 ** (self.refine_level_max - 1) / 2 ** (self.refine_level[blk] - 1))
            lo: float = self.block_bounds[blk, raxis, 0]
            ilo: int = np.argmin(np.abs(radius[:-1] - lo))

            _means: dict[str, NDArray] = {
                key: np.einsum("ijk->i", self.data(key)[blk, ...]) * vol_fracs[lb] for key in means.keys()
            }
            for i in range(nrb):
                jlo: int = ilo + i * lref_n
                jhi: int = ilo + (i + 1) * lref_n
                mapping[lb, i, :] = [jlo, jhi]
                for key in means.keys():
                    means[key][jlo:jhi] += _means[key][i]

        for k, v in means.items():
            _tmp = np.zeros_like(v)
            mpi.comm.Allreduce(v, _tmp, op=MPI.SUM)
            means[k] = _tmp / layer_volume

        for lb, blk in enumerate(blocklist):

            for i in range(self.ndim):
                vi: str = f"vel{axes[i]}"

                for j in range(i, self.ndim):
                    vj: str = f"vel{axes[j]}"

                    rs_key: str = f"R{axes[i]}{axes[j]}"

                    for rk in range(nrb):
                        for ii in range(mapping[lb, rk, 0], mapping[lb, rk, 1]):

                            stress[rs_key][ii] += (
                                np.sum(
                                    self._data["dens"][blk, rk, ...]
                                    * (self._data[vi][blk, rk, ...] - means[vi][ii])
                                    * (self._data[vj][blk, rk, ...] - means[vj][ii])
                                )
                                * vol_fracs[lb]
                            )

        for key in stress.keys():
            _tmp = np.zeros_like(stress[key])
            mpi.comm.Allreduce(stress[key], _tmp, op=MPI.SUM)
            stress[key][...] = _tmp / layer_volume

        return radius, stress, means

    def flame_window(self, radius: np.ndarray, stress: np.ndarray, mask: np.ndarray | None = None):

        super_gaussian = lambda x, amp, x0, sigma: amp * np.exp(-2 * ((x - x0) / sigma) ** 10)
        curve_fit = lambda func, x, y, p0: scipy.optimize.curve_fit(func, x, y, method="lm", p0=p0)

        # If mask provided apply it, else use an "all inclusive mask" for simplicity here
        ma_ = mask if mask is not None else np.where(radius < np.inf)[0]
        rd_ = radius[ma_]
        rs_ = {key: arr[ma_] for key, arr in stress.items()}

        # pind = rs_["Rxx"].argmax()
        # rpeak = rd_[pind]
        rind = np.where((rd_ <= np.inf))[0]
        rspan = rd_[rind]
        xfact = 1.0e5
        rspan /= xfact

        rsxx = rs_["Rxx"][rind]
        rfact = 10.0 ** np.max(np.floor(np.log10(rsxx)))
        rsxx /= rfact

        rmin = np.min(rspan)
        opt_xx, _ = curve_fit(
            super_gaussian,
            rspan - rmin,
            rsxx,
            (np.max(rsxx), rspan[np.argmax(rsxx)], np.std(rsxx)),
        )

        rsyyzz = (rs_["Ryy"][rind] + rs_["Rzz"][rind]) / rfact
        opt_yyzz, _ = curve_fit(
            super_gaussian,
            rspan - rmin,
            rsyyzz,
            (np.max(rsyyzz), rspan[np.argmax(rsyyzz)], np.std(rsyyzz)),
        )

        fit = opt_xx[1] + opt_yyzz[1]
        fit *= 0.5

        window_xmin = (fit - 16) * xfact
        window_xmax = (fit + 16) * xfact

        return window_xmin, window_xmax
