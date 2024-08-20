import copy
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np


from fava.mesh.unstructured import Unstructured
from fava.model import Model
from fava.util import timer


_field_mapping = {
    "tag": "id",
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
class FlashParticles(Unstructured):
    _filename = None
    _fields = {}
    _metadata_loaded = False


    def __init__(self, filename: Optional[str | Path] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename


    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename: str):
        fn_ = Path(filename)

        if fn_ is None:
            print("[WARNING] A filename has not been set, yet!")

        elif not (fn_.match("*hdf5_part_*") or fn_.match("*hdf5_chk_*")):
            raise Exception(f"[ERROR] FLASH datafiles with particles typically have 'hdf5_chk_' or 'hdf5_part_' in the filename: {fn_}!")

        elif fn_ != self._filename:
            self._metadata_loaded = False
            self._filename = fn_
            self._load_metadata()

    def _load_metadata(self):
        with h5py.File(self._filename, "r") as self._open_file:
            self._read_scalars()
            self._set_time_info()
            self._set_dimensionality()
            self._read_particles()

        self._metadata_loaded = True


    def _set_dimensionality(self):
        self.ndim = self._intscalars["dimensionality"]

    def _read_scalars(self):
        self._intscalars = {
            tpl[0].strip().decode("UTF-8"): tpl[1]
            for tpl in self._open_file["integer scalars"][()]
        }
        self._realscalars = {
            tpl[0].strip().decode("UTF-8"): tpl[1]
            for tpl in self._open_file["real scalars"][()]
        }
        # self._stringscalars = {
        #     tpl[0].decode("UTF-8").strip(): tpl[1].decode("UTF-8").strip()
        #     for tpl in self._open_file["string scalars"][()]
        # }


    def _read_particles(self):
        self.localnp = self._open_file["localnp"][()]
        self.nParticles = self._intscalars["globalnumparticles"]

        self._fields = [
            v.decode("UTF-8").strip()
            for v in np.squeeze(self._open_file["particle names"][()])
        ]

    def _load_particles(self, *args, **kwargs):
        """  """

        # Get the KWARGS
        fields_ = kwargs.get("fields", self._fields)
        ordered = kwargs.get("ordered", True)
        mask = kwargs.get("mask", None)

        self.data = {}

        with h5py.File(self.filename, "r") as h5file:

            _data = h5file["tracer particles"][()]
            for k, field in enumerate(fields_):
                if field not in self._fields:
                    print(
                        f"[WARNING] {field} particle field variable does not exist in dataset"
                    )
                    continue
                self.data[field] = _data[..., k]

            del _data

        if ordered:
            tidx = np.argsort(self.data["tag"])
            for field in self.data.keys():
                self.data[field] = self.data[field][tidx]

    def _set_time_info(self):
        self.dt = self._realscalars["dt"]
        self.dtold = self._realscalars["dtold"]
        self.time = self._realscalars["time"]

    def get_coords(self):

        coords = np.empty((self.nParticles, self.ndim))
        coords[:, 0] = self.data["posx"][:]

        print(self.ndim)

        if self.ndim > 1:
            coords[:, 1] = self.data["posy"][:]

        if self.ndim > 2:
            coords[:, 2] = self.data["posz"][:]

        return coords

