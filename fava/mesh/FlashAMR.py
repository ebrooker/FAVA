import h5py
import numpy as np

from functools import reduce
from pathlib import Path
from typing import Optional, List

from fava.geometry import GEOMETRY
from fava.temporary import Temporary
from fava.mesh import Structured


from enum import Enum, auto
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

class EDGE(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

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
    "vort": "vorticity"
}

@Temporary.register_mesh()
class FlashAMR(Structured):

    _filename  = None
    _metadata_loaded = False
    use_particles = False
    is_part_file = False
    nxb = 1
    nyb = 0
    nzb = 0
    xmin, ymin, zmin = 0.0, 0.0, 0.0
    xmax, ymax, zmax = 1.0, 1.0, 1.0


    def __init__(self, filename: Optional[str]=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

    
    @classmethod
    def is_this_your_mesh(self, filename: str | Path, *args, **kwargs):
        fn_types = ("hdf5_chk_", "hdf5_plt_cnt_", "hdf5_part_")
        return any(t in filename for t in fn_types)



    @property
    def filename(self):
        return self._filename
    
    @filename.setter
    def filename(self, filename):
        self.use_particles = "part_" in filename
        if filename is None:
            print("[WARNING] A filename has not been set, yet!")
        elif filename != self._filename:
            self._metadata_loaded = False
            self.is_part_file = "hdf5_part_" in filename
            self._filename = Path(filename)
            self._load_metadata()


    def _load_metadata(self):
        with h5py.File(self._filename, "r") as self._open_file:
            self.use_particles = "particle names" in self._open_file.keys()

            self._read_scalars()
            self._read_runtime_parameters()
            self._read_Nvars_list()
            self._set_time_info()
            self._set_dimensionality()
            self._set_geometry()
            self._set_blocks()
            self._set_domain()

            if self.use_particles:
                self._read_particles()
                self._set_particles()
            else:
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
|| {self.fields}
||_______________________________________________
||_______________________________________________
"""
        # print(flash_docstring)
        self._metadata_loaded = True

    def _read_particles(self):
        self._part_vars = [v.decode("UTF-8").strip() for v in np.squeeze(self._open_file["particle names"][()])]
        self.localnp = self._open_file["localnp"][()]

    def _set_particles(self):
        self.nParticles = self._intscalars["globalnumparticles"]


    def _read_scalars(self):
        self._intscalars    = {tpl[0].strip().decode("UTF-8") : tpl[1] for tpl in self._open_file["integer scalars"][()]}
        self._realscalars   = {tpl[0].strip().decode("UTF-8") : tpl[1] for tpl in self._open_file["real scalars"][()]}
        self._stringscalars = {tpl[0].decode("UTF-8").strip() : tpl[1].decode("UTF-8").strip() for tpl in self._open_file["string scalars"][()]}

    def _read_runtime_parameters(self):
        self._intrunpars  = {tpl[0].strip().decode("UTF-8") : tpl[1] for tpl in self._open_file["integer runtime parameters"][()]}
        self._realrunpars = {tpl[0].strip().decode("UTF-8") : tpl[1] for tpl in self._open_file["real runtime parameters"][()]}

    def _read_Nvars_list(self):
        self._flash_vars = [v.decode("UTF-8").strip() for v in np.squeeze(self._open_file["unknown names"][()])]
        self._nvars      = len(self._flash_vars)

    def _set_fields(self):
        self._fields = {}
        flash_fields = _field_mapping.keys()
        notmapped = "NotMapped"
        
        for field in self._flash_vars:
            
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

        if not self.is_part_file:
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

        if self.ndim > 1:
            self.ymin = self._realrunpars["ymin"]
            self.ymax = self._realrunpars["ymax"]
            self.yspn = self.ymax - self.ymin

        if self.ndim > 2:
            self.zmin = self._realrunpars["zmin"]
            self.zmax = self._realrunpars["zmax"]
            self.zspn = self.zmax - self.zmin

    def _set_time_info(self):
        self.dt    = self._realscalars["dt"]
        self.dtold = self._realscalars["dtold"]
        self.time  = self._realscalars["time"]


    def load(self, fields: Optional[List[str]]=None):
        
        fields_ = fields if fields is not None else self._flash_vars
        
        if not self.is_part_file:
            self.data = {}

        if self.use_particles:
            self.part_data = {}

        with h5py.File(self._filename, "r") as h5file:
            if self.use_particles:
                for k,field in enumerate(self._part_vars):
                    self.part_data[field] = h5file["tracer particles"][...,k]

            if not self.is_part_file:
                for field in fields_:
                    if field not in self._flash_vars:
                        print(f"[WARNING] {field} field variable does not exist in dataset!")
                        continue
                    self.data[field] = np.squeeze(np.swapaxes(
                        h5file[f"{field:4}"][()],
                        axis1 = 1,
                        axis2 = 3
                    ))


            
            # ToDo: Need to setup children and neighbor GID arrays
        
    @property
    def domain_volume(self) -> float:
        if self.geometry == GEOMETRY.CARTESIAN:
            vol = self.xmax - self.xmin
            if self.ndim > 1:
                vol *= self.ymax - self.ymin
            if self.ndim > 2:
                vol *= self.zmax - self.zmin

        return vol
    
    @property
    def cell_volume_max(self) -> float:
        return self.get_cell_volume_from_refinement()
    
    @property
    def cell_volume_min(self) -> float:
        return self.get_cell_volume_from_refinement(self.lrefmax)


    def get_point_data(self, blockID: int, point: List[int], field: str) -> float:

        match self.ndim:
            case 1:
                val = self.data[field][blockID,point[0]]
            case 2:
                val = self.data[field][blockID,point[0],point[1]]
            case 3:
                val = self.data[field][blockID,point[0],point[1],point[2]]
        return val


    def get_coord_index(self, point, block_list: List[int]) -> List[int]:
        idx = [None,None,None][:self.ndim]
        for blk in block_list:
            in_blk = self.is_point_in_block(point, blk)

            if in_blk:
                xcoord = self.get_cell_coords(0, blk)
                idx[0] = (np.abs(xcoord-point[0])).argmin()
                
                if self.ndim > 1:
                    ycoord = self.get_cell_coords(1, blk)
                    idx[1] = (np.abs(ycoord-point[1])).argmin()

                if self.ndim > 2:
                    zcoord = self.get_cell_coords(2, blk)
                    idx[2] = (np.abs(zcoord-point[2])).argmin()

                break

        return idx, blk


    def get_cell_volume_from_refinement(self, refine_level: int=0) -> float:
        if self.geometry == GEOMETRY.CARTESIAN:
            cells =  self.nxb * self.nblockx * refine_level
            if self.ndim > 1:
                cells *= self.nyb * self.nblocky * refine_level
            if self.ndim > 2:
                cells *= self.nzb * self.nblockz * refine_level

        return self.domain_volume / float(cells)


    def get_list_of_blocks(self, blkType: str="LEAF") -> List[int] | np.array:

        blkList = np.empty(self.nLocalBlocks, dtype=int)

        match BLOCK_TYPE[blkType]:

            case BLOCK_TYPE.LEAF:
                blkList = [
                    i for i in range(self.nBlocks)
                    if BLOCK_TYPE(self.blk_node[i]) == BLOCK_TYPE.LEAF
                ]

            case BLOCK_TYPE.ALL:
                blkList = [i for i in range(self.nBlocks)]

        return np.array(blkList, dtype=int)
    

    def get_cell_coords(self, axis: str, blockID: int, edge: str="CENTER", guardcell: bool=False):

        n = self.nCellsVec[axis]
        lb,ub = self.blk_bounds[blockID,axis,:]
        dx = (ub - lb) / float(n)

        m = n

        if guardcell:
            lb = lb - NGUARD*dx
            m += NGUARD

        match EDGE[edge]:
            case EDGE.CENTER:
                lb += 0.5*dx
                ub += 0.5*dx
            case EDGE.RIGHT:
                lb += dx
                ub += dx

        return np.linspace(lb,ub,m)


    def get_block_grid_coords(self, blockID: int, edge: str="CENTER", guardcell: bool=False):

        x = self.get_cell_coords(axis=0, blockID=blockID, edge=edge, guardcell=guardcell)

        if self.ndim == 2:
            y = self.get_cell_coords(axis=1, blockID=blockID, edge=edge, guardcell=guardcell)
            x,y = np.meshgrid(x, y, indexing='ij')
            x = np.array((x.flatten(), y.flatten()))

        elif self.ndim == 3:
            y = self.get_cell_coords(axis=1, blockID=blockID, edge=edge, guardcell=guardcell)
            z = self.get_cell_coords(axis=2, blockID=blockID, edge=edge, guardcell=guardcell)
            x,y,z = np.meshgrid(x, y, z, indexing='ij')
            x = np.array((x.flatten(), y.flatten(), z.flatten()))
        
        return x


    def get_block_physical_size(self, blockID: int) -> List[float]:
        return self.blk_size[blockID,:self.ndim]


    def get_cell_side_lengths(self, blockID):

        cells    = self.nCellsVec[:self.ndim]
        lengths  = self.get_block_physical_size(blockID)
        if self.geometry == GEOMETRY.CARTESIAN:
            lengths = [l/n for l,n in zip(lengths,cells)]

        return lengths

    def get_cell_volume(self, blockID):
        return self.get_cell_volume_from_refinement(self.blk_lref[blockID])


    def is_point_in_block(self, point: List[float]| np.array, blockID: int) -> bool:

        box = self.blk_bounds[blockID,...]

        is_in_box = (box[0,0] <= point[0] and point[0] < box[0,1])
        if self.ndim > 1:
            is_in_box = is_in_box and (box[1,0] <= point[1] and point[1] < box[1,1])
        if self.ndim > 2:
            is_in_box = is_in_box and (box[2,0] <= point[2] and point[2] < box[2,1])

        return is_in_box


    def get_particle_coords(self):

        coords = np.empty((self.nParticles,self.ndim))
        coords[:,0] = self.part_data["posx"][:]

        if self.ndim == 2:
            coords[:,1] = self.part_data["posy"][:]

        elif self.ndim == 3:
            coords[:,2] = self.part_data["posz"][:]
        
        return coords


    def uniformly_refine(self, refine_level: int):
        """
            This one is going to be fun!

            We need to take every block that is coarser than the specified uniform refinement level and break them up!
            The real question, should we also apply any interpolation to smooth the data? Probably not.

            Then it gets even more FUN... we have to find any blocks that are finer than the specified refinement level
            and derefine them...! 
        """
        pass

    def contiguous_volume(self, field, starting_point, cells):
        ...
        
