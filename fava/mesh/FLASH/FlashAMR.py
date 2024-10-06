import copy
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from functools import cached_property

import h5py
import numpy as np
import scipy.optimize
import yt

from fava.geometry import AXIS, EDGE, GEOMETRY
from fava.mesh.structured import Structured
from fava.model import Model

FLOAT = np.float64


def super_gaussian(x, amp, x0, sigma):
    return amp * np.exp(-2 * ((x - x0) / sigma) ** 10)


def curve_fit(func, x, y, p0):
    return scipy.optimize.curve_fit(func, x, y, method="lm", p0=p0)


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


NGUARD = 4

_field_mapping = {
    "velx": "velocity-x",
    "vely": "velocity-y",
    "velz": "velocity-z",
    "dens": "density",
    "pres": "pressure",
    "temp": "temperature",
    "ener": "energy",
    "flam": "flame progress",
    "igtm": "ignition time",
    "divv": "velocity-divergence",
    "vort": "vorticity",
}


@Model.register_mesh()
class FlashAMR(Structured):

    _filename = None
    _fields = {}
    _metadata_loaded = False
    nxb = 1
    nyb = 1
    nzb = 1
    xmin, ymin, zmin = 0.0, 0.0, 0.0
    xmax, ymax, zmax = 1.0, 1.0, 1.0

    def __init__(self, filename: Optional[str | Path] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    def __repr__(self) -> str:
        fstr = "\n".join(f"\t{val} --> {key}" for key, val in self._fields.items() if key != "NotMapped")
        fstr += f"\n\tNotMapped --> {self._fields.get('NotMapped')}"
        s = [
            "<FlashAMR>",
            f"\t{self._filename.name}; {self.ndim}D {self.geometry.name}",
            f"\ttime={self.time}, left_corner=({self.xmin},{self.ymin},{self.zmin}), right_corner=({self.xmax},{self.ymax},{self.zmax}))",
            f"{fstr}",
            "<FlashAMR/>",
        ]
        return "\n".join(s)

    @classmethod
    def is_this_your_mesh(self, filename: str, *args, **kwargs):
        fn_types = ("hdf5_chk_", "hdf5_plt_cnt_")
        return any(t in filename for t in fn_types)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename: Optional[str]):
        if filename is None:
            print("[WARNING] A filename has not been set, yet!")
        elif filename != self._filename:
            self._metadata_loaded = False
            self._filename = Path(filename)
            self._load_metadata()

    def _load_metadata(self):
        with h5py.File(self._filename, "r") as self._open_file:
            self._read_scalars()
            self._read_runtime_parameters()
            self._read_Nvars_list()
            self._set_time_info()
            self._set_dimensionality()
            self._set_geometry()
            self._set_blocks()
            self._set_domain()
            self._set_fields()

        flash_docstring = f"""
||-----------------------------------------------
||-----------------------------------------------
|| FLASH AMR Dataset
||-----------------------------------------------
|| Filename: {self._filename.name}
|| Location: {self._filename.parent}
||-----------------------------------------------
|| Current Time  = {self.time}
|| Next Timestep = {self.dt}
|| Last Timestep = {self.dtold}
||-----------------------------------------------
|| Dimensionality  = {self.ndim}
|| Domain Geometry = {self.geometry.name}
|| Lower Boundary  = {[self.xmin,self.ymin,self.zmin][:self.ndim]}
|| Upper Boundary  = {[self.xmax,self.ymax,self.zmax][:self.ndim]}
||-----------------------------------------------
|| Blocksize per axis  = {self.nCellsVec[:self.ndim]}
|| Blocksize in total  = {self.nBlkCells}
|| Min Blocks per axis = {self.nBlksVec[:self.ndim]}
|| Total Block Number  = {self.nBlocks}
||-----------------------------------------------
|| Available fields [API -> FlashAMR]
|| {self._fields}
||_______________________________________________
||_______________________________________________
"""
        # print(flash_docstring)
        self._metadata_loaded = True

    def _read_scalars(self):
        self._intscalars = {tpl[0].strip().decode("UTF-8"): tpl[1] for tpl in self._open_file["integer scalars"][()]}
        self._realscalars = {tpl[0].strip().decode("UTF-8"): tpl[1] for tpl in self._open_file["real scalars"][()]}
        self._stringscalars = {
            tpl[0].decode("UTF-8").strip(): tpl[1].decode("UTF-8").strip() for tpl in self._open_file["string scalars"][()]
        }

    def _read_runtime_parameters(self):
        self._intrunpars = {tpl[0].strip().decode("UTF-8"): tpl[1] for tpl in self._open_file["integer runtime parameters"][()]}
        self._realrunpars = {tpl[0].strip().decode("UTF-8"): tpl[1] for tpl in self._open_file["real runtime parameters"][()]}

    def _read_Nvars_list(self):
        self._flash_fields = [v.decode("UTF-8").strip() for v in np.squeeze(self._open_file["unknown names"][()])]
        self._nvars = len(self._flash_fields)

    def _set_fields(self):
        self._fields = {}
        flash_fields = _field_mapping.keys()
        notmapped = "NotMapped"

        for field in self._flash_fields:

            if field in flash_fields:
                self._fields[_field_mapping[field]] = field

            elif notmapped in self._fields:
                self._fields[notmapped].append(field)

            else:
                self._fields[notmapped] = [field]

    def _set_dimensionality(self):
        self.ndim = self._intscalars["dimensionality"]

    def _set_blocks(self):
        # Get block lengths and total size
        self.nxb = self._intscalars["nxb"]
        self.nBlkCells = self.nxb
        if self.ndim > 1:
            self.nyb = self._intscalars["nyb"]
            self.nBlkCells *= self.nyb

        if self.ndim > 2:
            self.nzb = self._intscalars["nzb"]
            self.nBlkCells *= self.nzb

        self.nCellsVec = [self.nxb, self.nyb, self.nzb]  # Vector of block lengths

        if "total blocks" in self._intscalars:
            self.nBlocks = self._intscalars["total blocks"]
        else:
            self.nBlocks = self._intscalars["globalnumblocks"]

        # ToDo: Ensure this is truly parallelized
        self.nLocalBlocks = self.nBlocks

        self.iprocs = self._intscalars["iprocs"]
        self.jprocs = self._intscalars["jprocs"]
        self.kprocs = self._intscalars["kprocs"]

        self.lrefmax = self._intrunpars["lrefine_max"]
        self.nblockx = self._intrunpars["nblockx"]
        self.nblocky = self._intrunpars["nblocky"]
        self.nblockz = self._intrunpars["nblockz"]
        self.nBlksVec = [self.nblockx, self.nblocky, self.nblockz]

        self.blk_size = self._open_file["block size"][()]
        self.blk_bounds = self._open_file["bounding box"][()]
        self.blk_node = self._open_file["node type"][()]
        self.blk_lref = self._open_file["refine level"][()]
        self.gid = self._open_file["gid"][()]

        # Now we need to get some simple values
        self.max_refine_level = self.blk_lref.max()

    def _set_geometry(self):
        self.geometry = GEOMETRY[self._stringscalars["geometry"].upper()]

    def _set_domain(self):
        # Set the domain boundaries
        self.xmin = self._realrunpars["xmin"]
        self.xmax = self._realrunpars["xmax"]
        self.xspn = self.xmax - self.xmin
        domain_bounds = [[self.xmin, self.xmax]]

        if self.ndim > 1:
            self.ymin = self._realrunpars["ymin"]
            self.ymax = self._realrunpars["ymax"]
            self.yspn = self.ymax - self.ymin
            domain_bounds.append([self.ymin, self.ymax])

        if self.ndim > 2:
            self.zmin = self._realrunpars["zmin"]
            self.zmax = self._realrunpars["zmax"]
            self.zspn = self.zmax - self.zmin
            domain_bounds.append([self.zmin, self.zmax])

        self.domain_bounds = np.array(domain_bounds)

    def _set_time_info(self):
        self.dt = self._realscalars["dt"]
        self.dtold = self._realscalars["dtold"]
        self.time = self._realscalars["time"]

    def _load_mesh(self, fields: Optional[List[str]] = None):

        fields_ = fields if fields is not None else self._flash_fields

        self.data = {}
        for field in fields_:
            if field not in self._flash_fields:
                print(f"[WARNING] {field} field variable does not exist in dataset!")
                continue
            self.data[field] = np.ascontiguousarray(np.swapaxes(self._h5file[f"{field:4}"][()], axis1=1, axis2=3).astype(FLOAT))

        self.data_1d_view = {}
        for field in self.data.keys():
            self.data_1d_view[field] = self.data[field].view().reshape(-1)
        # ToDo: Need to setup children and neighbor GID arrays

    def load(self, fields: Optional[List[str]] = None):
        with h5py.File(self._filename, "r") as self._h5file:
            self._load_mesh(fields)

    @cached_property
    def domain_volume(self) -> float:
        if self.geometry == GEOMETRY.CARTESIAN:
            vol = self.xmax - self.xmin
            if self.ndim > 1:
                vol *= self.ymax - self.ymin
            if self.ndim > 2:
                vol *= self.zmax - self.zmin

        return vol

    @cached_property
    def cell_volume_max(self) -> float:
        return self.get_cell_volume_from_refinement()

    @cached_property
    def cell_volume_min(self) -> float:
        return self.get_cell_volume_from_refinement(self.lrefmax)

    @cached_property
    def cell_volumes(self) -> np.ndarray:
        blk_list = self.get_list_of_blocks()
        volumes = np.zeros_like(self.blk_lref)
        for lb in blk_list:
            volumes[lb] = self.get_cell_volume(lb)
        return volumes

    def get_point_data(self, blockID: int, point: List[int], field: str) -> float:

        match self.ndim:
            case 1:
                val = self.data[field][blockID, point[0]]
            case 2:
                val = self.data[field][blockID, point[0], point[1]]
            case 3:
                val = self.data[field][blockID, point[0], point[1], point[2]]
        return val

    def get_coord_index(self, point, block_list: List[int]) -> Tuple[List, int]:
        idx = [None, None, None][: self.ndim]
        for blk in block_list:
            in_blk = self.is_point_in_block(point, blk)

            if in_blk:
                xcoord = self.get_cell_coords(0, blk)
                idx[0] = (np.abs(xcoord - point[0])).argmin()

                if self.ndim > 1:
                    ycoord = self.get_cell_coords(1, blk)
                    idx[1] = (np.abs(ycoord - point[1])).argmin()

                if self.ndim > 2:
                    zcoord = self.get_cell_coords(2, blk)
                    idx[2] = (np.abs(zcoord - point[2])).argmin()

                break

        return idx, blk

    def get_cell_volume_from_refinement(self, refine_level: int = 1) -> float:
        if self.geometry == GEOMETRY.CARTESIAN:
            cells = self.nxb * self.nblockx * 2 ** (refine_level - 1)
            if self.ndim > 1:
                cells *= self.nyb * self.nblocky * 2 ** (refine_level - 1)
            if self.ndim > 2:
                cells *= self.nzb * self.nblockz * 2 ** (refine_level - 1)
        return self.domain_volume / float(cells)

    def get_list_of_blocks(self, blkType: str = "LEAF") -> List[int] | np.array:

        match BLOCK_TYPE[blkType]:

            case BLOCK_TYPE.LEAF:
                blkList = [i for i in range(self.nBlocks) if BLOCK_TYPE(self.blk_node[i]) == BLOCK_TYPE.LEAF]

            case BLOCK_TYPE.ALL:
                blkList = [i for i in range(self.nBlocks)]

        return np.array(blkList, dtype=int)

    def get_minimum_deltas(self, axis: int):
        return (self.domain_bounds[axis, 1] - self.domain_bounds[axis, 0]) / (
            self.nCellsVec[axis] * self.nBlksVec[axis] * 2 ** (self.lrefmax - 1)
        )

    def get_maximum_deltas(self, axis: int):
        return (self.domain_bounds[axis, 1] - self.domain_bounds[axis, 0]) / (
            self.nCellsVec[axis] * self.nBlksVec[axis] * 2 ** (self.blk_lref.min() - 1)
        )

    def get_deltas_from_refine_level(self, refine_level: int) -> List[float]:
        _res = []
        for i in range(self.ndim):
            _res.append(self.get_delta_from_refine_level(axis=i, refine_level=refine_level))
        return _res

    def get_delta_from_refine_level(self, axis: int, refine_level: int) -> float:
        return (self.domain_bounds[axis, 1] - self.domain_bounds[axis, 0]) / (
            self.nCellsVec[axis] * self.nBlksVec[axis] * 2 ** (refine_level - 1)
        )

    def get_block_deltas(self, blockID: int) -> List[float]:
        _res = []
        for i in range(self.ndim):
            _res.append(self.get_block_delta(axis=i, blockID=blockID))
        return _res

    def get_block_delta(self, axis: int, blockID: int) -> float:
        return (self.blk_bounds[blockID, axis, 1] - self.blk_bounds[blockID, axis, 0]) / (self.nCellsVec[axis])

    def get_block_bounds(self, axis: int, blockID: int):
        return self.blk_bounds[blockID, axis, :]

    def get_cell_coords(self, axis: int, blockID: int, edge: str = "CENTER", guardcell: bool = False):

        n = self.nCellsVec[axis]
        lb, ub = self.blk_bounds[blockID, axis, :]
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

    def get_block_grid_coords(self, blockID: int, edge: str = "CENTER", guardcell: bool = False):

        x = self.get_cell_coords(axis=0, blockID=blockID, edge=edge, guardcell=guardcell)

        if self.ndim == 2:
            y = self.get_cell_coords(axis=1, blockID=blockID, edge=edge, guardcell=guardcell)
            x, y = np.meshgrid(x, y, indexing="ij")
            x = np.array((x.flatten(), y.flatten()))

        elif self.ndim == 3:
            y = self.get_cell_coords(axis=1, blockID=blockID, edge=edge, guardcell=guardcell)
            z = self.get_cell_coords(axis=2, blockID=blockID, edge=edge, guardcell=guardcell)
            x, y, z = np.meshgrid(x, y, z, indexing="ij")
            x = np.array((x.flatten(), y.flatten(), z.flatten()))

        return x

    def get_block_physical_size(self, blockID: int) -> List[float]:
        return self.blk_size[blockID, : self.ndim]

    def get_cell_side_lengths(self, blockID):

        cells = self.nCellsVec[: self.ndim]
        lengths = self.get_block_physical_size(blockID)
        if self.geometry == GEOMETRY.CARTESIAN:
            lengths = [l / n for l, n in zip(lengths, cells)]

        return lengths

    def get_cell_volume(self, blockID):
        return self.get_cell_volume_from_refinement(self.blk_lref[blockID])

    def points_within_block(
        self,
        points: List[float] | np.array,
        axis: int,
        blockID: int,
        return_indices=False,
    ):
        box = self.blk_bounds[blockID, axis, :]
        if isinstance(points, list):
            points_ = np.array(points)
        else:
            points_ = np.copy(points)

        in_indices = np.where((points_ >= box[0]) & (points_ <= box[1]))

        if return_indices:
            return points_[in_indices], in_indices
        else:
            return points_[in_indices]

    def is_point_in_block(self, point: List[float] | np.array, blockID: int) -> bool:

        box = self.blk_bounds[blockID, ...]

        is_in_box = box[0, 0] <= point[0] and point[0] < box[0, 1]
        if self.ndim > 1:
            is_in_box = is_in_box and (box[1, 0] <= point[1] and point[1] < box[1, 1])
        if self.ndim > 2:
            is_in_box = is_in_box and (box[2, 0] <= point[2] and point[2] < box[2, 1])

        return is_in_box

    def refine_to_finest(
        self,
        refine_level: Optional[int] = None,
        subdomain_bounds: Optional[list | np.ndarray] = None,
    ):
        """
        This one is going to be fun!

        We need to take every block that is coarser than the specified uniform refinement level and break them up!
        The real question, should we also apply any interpolation to smooth the data? Probably not.

        Then it gets even more FUN... we have to find any blocks that are finer than the specified refinement level
        and derefine them...!
        """

        subdomain = True if subdomain_bounds is not None else False

        if subdomain:
            unibounds = np.squeeze(np.array(subdomain_bounds))
        else:
            unibounds = self.domain_bounds

        if subdomain:
            msg = "[Subdomain Boundary Error]"
            if unibounds[0, 0] < self.xmin or self.xmax < unibounds[0, 1]:
                msg += "Subdomain x-coordinates exceed domain coordinates:\n"
                msg += f"\t Chosen subdomain x-coordinate range: ({unibounds[0,0]}, {unibounds[0,1]})\n"
                msg += f"\t Domain x-coordinate range: ({self.xmin}, {self.xmax})\n"
                raise Exception(msg)

            if self.ndim > 1 and (unibounds[1, 0] < self.ymin or self.ymax < unibounds[1, 1]):
                msg += "Subdomain y-coordinates exceed domain coordinates:\n"
                msg += f"\t Chosen subdomain y-coordinate range: ({unibounds[1,0]}, {unibounds[1,1]})\n"
                msg += f"\t Domain y-coordinate range: ({self.ymin}, {self.ymax})\n"
                raise Exception(msg)

            if self.ndim > 2 and (unibounds[2, 0] < self.zmin or self.zmax < unibounds[2, 1]):
                msg += "Subdomain z-coordinates exceed domain coordinates:\n"
                msg += f"\t Chosen subdomain z-coordinate range: ({unibounds[2,0]}, {unibounds[2,1]})\n"
                msg += f"\t Domain z-coordinate range: ({self.zmin}, {self.zmax})\n"
                raise Exception(msg)

        if subdomain:
            print(f"(xmin, xmax): ({self.xmin},{self.xmax}) --> ({unibounds[0,0]},{unibounds[0,1]})")
            if self.ndim > 1:
                print(f"(ymin, ymax): ({self.ymin},{self.ymax}) --> ({unibounds[1,0]},{unibounds[1,1]})")
                print(f"(zmin, zmax): ({self.zmin},{self.zmax}) --> ({unibounds[2,0]},{unibounds[2,1]})")
        # nblkx, nblky, nblkz = self.nBlksVec

        lref = self.lrefmax if refine_level is None else refine_level

        # Get zonal dimensions of refined grid
        lrefcells = 2 ** (lref - 1)
        dims = np.array(
            [nb * bl * lrefcells for nb, bl in zip(self.nCellsVec[: self.ndim], self.nBlksVec[: self.ndim])],
            dtype=int,
        )

        # Get the zone deltas (edge to)
        grid_delta = (unibounds[:, 1] - unibounds[:, 0]) / (float(dims) - 1.0)

        # Get the zone half deltas (center to edge)
        grid_half_delta = 0.5e0 * grid_delta

        # Figure out the local block IDs first
        # ????

        fblk_x = self.nblockx * lrefcells
        fblk_y = self.nblocky * lrefcells
        fblk_z = self.nblockz * lrefcells

        # compute subdomain_cells here

        x = np.linspace(self.xmin, self.xmax - dx2, dims[0])
        if self.ndim > 1:
            dy = (self.ymax - self.ymin) / float(dims[1])
            y = np.linspace(self.ymin, self.ymax - dy, dims[1])
        if self.ndim > 2:
            dz = (self.zmax - self.zmin) / float(dims[2])
            z = np.linspace(self.zmin, self.zmax - dz, dims[2])

        cdx = np.zeros((self.nBlocks, 3), dtype=int)
        for cidx in range(cdx.shape[0]):
            cdx[cidx, 0] = np.argmin(np.abs(x - self.blk_bounds[cidx, 0, 0]))

        if self.ndim > 1:
            for cidx in range(cdx.shape[0]):
                cdx[cidx, 1] = np.argmin(np.abs(y - self.blk_bounds[cidx, 1, 0]))

        if self.ndim > 2:
            for cidx in range(cdx.shape[0]):
                cdx[cidx, 2] = np.argmin(np.abs(z - self.blk_bounds[cidx, 2, 0]))

        indices = np.zeros((self.nBlocks, self.nBlkCells, 3, 2))

    def contiguous_volume(self, field: str, starting_point, cells): ...

    def rms(self, field: str, particles: bool = False):
        """Returns the root mean square (RMS) of the field provided (pass the parameter <particles=True> to use particle data instead of mesh data)"""
        if particles:
            return np.sqrt(np.mean(self.particles[field] ** 2))
        else:
            return np.sqrt(np.mean(self.data[field] ** 2))

    def add_field(self, name: str, expression: str, particles: Optional[bool] = False):

        expr_ = copy.copy(expression)

        n = 0
        if particles:
            for field in self.particles.keys():
                if field in expr_:
                    expr_ = expr_.replace(field, f"self.particles['{field}']")
                    n += 1
            if n > 0:
                self.particles[name] = eval(expr_)
        else:
            for field in self.data.keys():
                if field in expr_:
                    expr_ = expr_.replace(field, f"self.data['{field}']")
                    n += 1
            if n > 0:
                self.data[name] = eval(expr_)

    def slice_average(self, field: str, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
        field_: str | None = self._fields.get(field)
        if field_ is None:
            field_ = field

        ax_ = AXIS(axis)

        min_deltas = [self.get_minimum_deltas(ii) for ii in range(self.ndim)]

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
        field_: str | None = self._fields.get(field)
        if field_ is None:
            field_ = field

        lrefcells: int = 2 ** (self.lrefmax - 1)
        dims: list = [nb * bl * lrefcells for nb, bl in zip(self.nCellsVec[: self.ndim], self.nBlksVec[: self.ndim])]

        ax_ = AXIS(axis)

        min_deltas = [self.get_minimum_deltas(ii) for ii in range(self.ndim)]

        match ax_:
            case AXIS.I:
                rmin, rmax = self.xmin, self.xmax
                nrb = self.nxb
            case AXIS.J:
                rmin, rmax = self.ymin, self.ymax
                nrb = self.nyb
            case AXIS.K:
                rmin, rmax = self.zmin, self.zmax
                nrb = self.nzb
            case _:
                raise ValueError(f"Do not recognize AXIS enumeration {ax_}")

        span = np.linspace(rmin, rmax, dims[ax_.value] + 1, dtype=np.float64)

        blocklist = self.get_list_of_blocks()
        alp = np.zeros(dims[ax_.value], dtype=np.float64)
        for blkID in blocklist:
            lref_n = int(2 ** (self.lrefmax - 1) / 2 ** (self.blk_lref[blkID] - 1))
            lo, _ = self.blk_bounds[blkID, 0, :]
            ilo = np.argmin(np.abs(span[:-1] - lo))

            dvol_red = min_deltas[ax_.value] / self.get_delta_from_refine_level(ax_, self.blk_lref[blkID])
            volFrac = self.get_cell_volume(blkID) * dvol_red
            for i in range(nrb):
                jlo = ilo + i * lref_n
                jhi = ilo + (i + 1) * lref_n
                for j in range(jlo, jhi):
                    alp[j] += np.sum(self.data[field_][blkID, i, ...]) * volFrac

        return span, alp

    def volume_average(self, field: str) -> float:
        field_ = self._fields.get(field)
        if field_ is None:
            field_ = field

        blk_list = self.get_list_of_blocks()
        return np.mean(self.data[field_][blk_list, ...] * self.volumes[blk_list, None, None, None] / self.domain_volume)

    def volume_integration(self, field: str):
        field_ = self._fields.get(field)
        if field_ is None:
            field_ = field

        blk_list = self.get_list_of_blocks()
        return np.sum(self.data[field_][blk_list, ...] * self.volumes[blk_list, None, None, None] / self.domain_volume)

    def pdf1d(self, field: str, *args, **kwargs):
        raise NotImplementedError

    def pdf2d(self, field: str, *args, **kwargs):
        raise NotImplementedError

    def structure_functions(self, *args, **kwargs):
        raise NotImplementedError

    def fractal_dimension(self, *args, **kwargs):
        raise NotImplementedError

    def multi_fractal_dimension(self, *args, **kwargs):
        raise NotImplementedError

    def velocity_statistics(self, *args, **kwargs):
        raise NotImplementedError

    def reynolds_stress(self, raxis: int = 0) -> tuple[np.ndarray, np.ndarray]:
        lrefcells: int = 2 ** (self.lrefmax - 1)
        dims = [nb * bl * lrefcells for nb, bl in zip(self.nCellsVec[: self.ndim], self.nBlksVec[: self.ndim])]

        ax_ = AXIS(raxis)

        min_deltas = [self.get_minimum_deltas(ii) for ii in range(self.ndim)]

        axes = "xyz"[: self.ndim]
        match ax_:
            case AXIS.I:
                layer_volume: float = (self.ymax - self.ymin) * (self.zmax - self.zmin)
                rmin, rmax = self.xmin, self.xmax
                nrb = self.nxb
            case AXIS.J:
                layer_volume: float = (self.xmax - self.xmin) * (self.zmax - self.zmin)
                rmin, rmax = self.ymin, self.ymax
                nrb = self.nyb
            case AXIS.K:
                layer_volume: float = (self.ymax - self.ymin) * (self.xmax - self.xmin)
                rmin, rmax = self.zmin, self.zmax
                nrb = self.nzb
            case _:
                raise ValueError(f"Do not recognize AXIS enumeration {ax_}")

        layer_volume *= min_deltas[ax_.value]

        dr = (rmax - rmin) / float(dims[ax_.value])
        radius = np.linspace(rmin, rmax, dims[ax_.value] + 1)

        stresses = {}
        means = {"dens": np.zeros(dims[ax_.value])}
        for i in range(self.ndim):
            means[f"vel{axes[i]}"] = np.zeros(dims[ax_.value])
            for j in range(i, self.ndim):
                stresses[f"R{axes[i]}{axes[j]}"] = np.zeros(dims[ax_.value])

        blocklist = self.get_list_of_blocks()

        bindices = np.zeros((blocklist.size, nrb, 2), dtype=int)
        for lb, blkID in enumerate(blocklist):
            lref_n = int(2 ** (self.lrefmax - 1) / 2 ** (self.blk_lref[blkID] - 1))
            lo, _ = self.blk_bounds[blkID, 0, :]
            ilo = np.argmin(np.abs(radius[:-1] - lo))

            dvol_red = min_deltas[ax_.value] / self.get_delta_from_refine_level(ax_.value, self.blk_lref[blkID])
            volFrac = self.get_cell_volume(blkID) * dvol_red
            for i in range(nrb):
                bindices[lb, i, 0] = ilo + i * lref_n
                bindices[lb, i, 1] = ilo + (i + 1) * lref_n
                jlo, jhi = bindices[lb, i, :]
                for key in means.keys():
                    for j in range(jlo, jhi):
                        means[key][j] += np.sum(self.data[key][blkID, i, ...]) * volFrac

        means = {k: v / layer_volume for k, v in means.items()}

        for lb, blkID in enumerate(blocklist):
            dvol_red = min_deltas[ax_.value] / self.get_delta_from_refine_level(ax_.value, self.blk_lref[blkID])
            volFrac = self.get_cell_volume(blkID) * dvol_red

            for xk in range(nrb):
                ilo, ihi = bindices[lb, xk, :]
                dens = self.data["dens"][blkID, xk, ...]

                for i in range(self.ndim):
                    vi = f"vel{axes[i]}"
                    veli = self.data[vi][blkID, xk, ...]

                    for j in range(i, self.ndim):
                        vj = f"vel{axes[j]}"
                        velj = self.data[vj][blkID, xk, ...]

                        RSkey = f"R{axes[i]}{axes[j]}"

                        for k in range(ilo, ihi):
                            stresses[RSkey][k] += np.sum(dens * (veli - means[vi][k]) * (velj - means[vj][k])) * volFrac

        for key in stresses.keys():
            stresses[key] /= layer_volume * means["dens"]

        return radius, stresses

    def flame_window(self, radius: np.ndarray, stress: np.ndarray, mask: np.ndarray | None = None):

        # If mask provided apply it, else use an "all inclusive mask" for simplicity here
        ma_ = mask if mask is not None else np.where(radius < np.inf)[0]
        rd_ = radius[ma_]
        rs_ = {key: arr[ma_] for key, arr in stress.items()}

        pind = rs_["Rxx"].argmax()

        rpeak = rd_[pind]
        rind = np.where((rd_ <= np.inf))[0]  # np.where( (rpeak <= rd_) & (rd_ <= (rpeak + 64e5)) )
        rspan = rd_[rind]
        xfact = 1.0e5
        rspan /= xfact

        rmean = np.mean(rspan)
        rsxx = rs_["Rxx"][rind]
        rfact = 10.0 ** np.max(np.floor(np.log10(rsxx)))
        rsxx /= rfact

        rmin = np.min(rspan)
        rmax = np.max(rspan)
        opt_xx, _ = curve_fit(
            super_gaussian,
            rspan - rmin,
            rsxx,
            (np.max(rsxx), rspan[np.argmax(rsxx)], np.std(rsxx)),
        )

        rsyyzz = (rs_["Ryy"][rind] + rs_["Rzz"][rind]) / rfact
        rmax = np.max(rspan)
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

    def overwrite_velocities(self, xmin: float, xmax: float, filename: str):

        ds = yt.load(filename)

        domain_delta = ds.domain_right_edge.d - ds.domain_right_edge.d
        min_dims = ds.domain_dimensions
        max_dims = np.copy(min_dims)
        max_dims[: self.ndim] *= 2**ds.index.max_level

        min_delta = domain_delta / max_dims

        ds.close()

        left = np.array([xmin, self.ymin, self.zmin])
        right = np.array([xmax, self.ymax, self.zmax])
        domain = right - left
        dimensions = domain / min_delta

        ds = yt.load(self.filename)
        cube = ds.covering_grid(self.lrefmax, left_edge=ds.domain_left_edge.d, dims=dimensions)
