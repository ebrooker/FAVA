import copy
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from fava.model import FLASH
from fava.util import timer
from fava.util._mpi import mpi, FAVAInterruptHandler

CWD: Path = Path.cwd()
PIPELINE_CHECKPOINT_FILE: Path = CWD / "fava.checkpoint"
PIPELINE_SETTINGS_FILE: Path = CWD / "pipeline_settings.json"

LOGGER: logging.Logger = logging.getLogger(__file__)


class Pipeline:

    def __init__(self) -> None:
        self.checkpoint_data: dict = {}

    def load_settings(self, settings_path: Path) -> None:
        with settings_path.open("r") as f:
            self.settings: dict = json.load(f)

        self.checkpoint_data["settings"] = copy.deepcopy(self.settings)
        # Get model particulars, file basename, dimensionality, model name
        self.basename: str = self._get_validated_dictitem("basename", str, self.settings)
        self.ndim: int = self._get_validated_dictitem("dimension", int, self.settings)
        self.model: str = self._get_validated_dictitem("model", str, self.settings)

        # Format data directory
        self.data_dir: Path = Path(self._get_validated_dictitem("data folder", str, self.settings))

        # Format output directory
        self.output_dir: Path = Path(self._get_validated_dictitem("output folder", str, self.settings))

        self.model: FLASH = FLASH(self.data_dir)

    def _get_validated_dictitem(self, key, vtype, dictionary):
        assert key in dictionary
        assert isinstance(dictionary[key], vtype)
        return dictionary[key]

    def _flam_or_rpv1(self) -> bool:
        self.flam: str = "rpv1"
        if self.model.mesh.data(self.flam) is None:
            self.flam = "flam"

        if self.model.mesh.data(self.flam) is None:
            return False

        return True

    def checkpoint(self) -> None:
        if mpi.root:
            with PIPELINE_CHECKPOINT_FILE.open("w") as f:
                json.dump(self.checkpoint_data, f, ensure_ascii=True, indent=4)

    def restart(self) -> None:
        if PIPELINE_CHECKPOINT_FILE.is_file():
            with PIPELINE_CHECKPOINT_FILE.open("r") as f:
                self.checkpoint_data = json.load(f)

        self.load_settings(settings_path=PIPELINE_SETTINGS_FILE)

    def refresh_model(self) -> None:
        del self.model
        self.model = FLASH(self.data_dir)

    def reynolds_stress(self, index: int) -> None:

        file_type: str = "plt"
        self.model.load(file_index=index, file_type=file_type)

        fn: Path = self.output_dir / self.model.convert_filename_type(file_type, "anl").stem

        if mpi.root:
            print("REYNOLDS STRESS: ", fn, flush=True)

        try:
            pkey: str = "reynolds stresses"
            skey: str = "scalars"
            with h5py.File(fn, "r") as f:
                x = f[pkey]["radius"][()]
                s: dict = {rkey: f[pkey]["tensor"][rkey][()] for rkey in f[pkey]["tensor"].keys()}
        except:
            x, s, m = self.model.reynolds_stress()
            if mpi.root:
                self.model.save_to_hdf5(
                    data={"reynolds stresses": {"tensor": s, "radius": x, "means": m}}, filename=fn
                )

        success: bool = self._flam_or_rpv1()
        if not success:
            return

        span, alp = self.model.slice_average(self.flam, axis=0)
        ccspan: NDArray = 0.5 * (span[1:] + span[:-1])

        ccx: NDArray = 0.5 * (x[1:] + x[:-1])

        mask: NDArray = np.argwhere((0.0 < alp) & (alp < 1.0)).flatten()

        centroid: float = self.model.mesh.flame_window(ccx, s, mask)

        xmin: float = centroid - 16e5
        xmax: float = centroid + 16e5
        
        left: NDArray = self.model.mesh.domain_bounds[:, 0]
        right: NDArray = self.model.mesh.domain_bounds[:, 1]

        left[0] = xmin
        right[0] = xmax

        dx: float = 0.0
        if self.settings.get("flame window") is not None:
            dx = self.settings["flame window"].get("dx", 0.0)

        left[0] += dx
        right[0] += dx

        window_bounds = right - left
        window_dimensions = (window_bounds / self.model.mesh.get_minimum_deltas(axis=1)).astype(int)

        if mpi.root:
            print("Flame Window: ", right, window_dimensions, flush=True)

            self.model.save_to_hdf5(
                data={
                    skey: {
                        "time": self.model.mesh.time,
                        "window left": left,
                        "window right": right,
                        "window dimensions": window_dimensions,
                    }
                },
                filename=fn,
            )

    def smooth_window_trajectory(self) -> None:

        self.xmax: NDArray = np.zeros(self.model.nfiles(file_type="plt"))
        self.time: NDArray = np.zeros_like(self.xmax)

        for i, p in enumerate(sorted(self.model.plt_files["by index"].keys())):
            self.model.load(file_index=p, file_type="plt")

            fn: Path = self.output_dir / self.model.convert_filename_type("plt", "anl").stem

            with h5py.File(fn, "r") as f:
                win_right = f["scalars"]["window right"][()]

            self.xmax[i] = win_right[0]
            self.time[i] = self.model.mesh.time

        coef: NDArray = np.polyfit(self.time, self.xmax, 1)
        self.t0: float = self.time[0]
        self.x0: float = self.xmax[0]
        self.func = np.poly1d(coef)

    def extract_windows(self, index: int) -> None:
        self.model.load(file_index=index, file_type="plt")

        success: bool = self._flam_or_rpv1()
        if not success:
            return

        xmax = self.x0 + (self.func(self.model.mesh.time) - self.func(self.t0))
        subdomain_coords: NDArray = np.array([[xmax - 32e5, xmax], [-16e5, 16e5], [-16e5, 16e5]])
        fields: list[str] = [self.flam, "dens", "pres", "temp", "velx", "vely", "velz", "divv", "igtm", "vort"]

        fn: Path = self.output_dir / self.model.convert_filename_type("plt", "uni").stem

        if mpi.root:
            print("EXTRACT: ", fn, flush=True)

        if fn.is_file():
            return

        self.model.mesh.from_amr(subdomain_coords=subdomain_coords, fields=fields, filename=fn)

    def analyze_uniform_data(self, index: int) -> None:
        pkey: str = "analyze uniform data"
        self.model.load(file_index=index, file_type="uni")

        success: bool = self._flam_or_rpv1()
        if not success:
            return

        fn: Path = self.output_dir / self.model.convert_filename_type("uni", "anl").stem

        if mpi.root:
            print("ANALYSIS: ", fn, flush=True)

        analyses: dict = {
            "fractal dimension": self.model.fractal_dimension,
            "structure functions": self.model.structure_functions,
            "kinetic energy spectra": self.model.kinetic_energy_spectra,
        }

        akeys: list[str] = list(analyses.keys())
        begin_key: str = self.checkpoint_data[pkey].get("analysis")
        begin: int = 0
        if begin_key is not None and begin_key in akeys:
            begin = akeys.index(begin_key)

        for akey in list(analyses.keys())[begin:]:
            self.checkpoint_data[pkey]["analysis"] = akey

            if not self.settings[akey].get("skip", False):
                _settings: dict = self.settings[akey].get("settings", {})
                retval: dict = analyses[akey](**_settings)
                mpi.comm.barrier()
                if mpi.root:
                    self.model.save_to_hdf5(data={akey: retval}, filename=fn)
                mpi.comm.barrier()

        self.checkpoint_data[pkey]["analysis"] = None


@timer
def main() -> None:

    pipe = Pipeline()
    pipe.restart()

    if mpi.root:
        print("\n-------------\n", pipe.checkpoint_data, "\n-------------\n", flush=True)

    with FAVAInterruptHandler(external_handler=pipe.checkpoint) as fih:
        pkey: str = "reynolds stress"
        if not pipe.settings[pkey].get("skip", False):

            rdict: dict = pipe.checkpoint_data.get(pkey, {})
            begin: int = rdict.get("index", 0)

            for i in sorted(pipe.model.plt_files["by index"].keys())[begin:]:
                pipe.reynolds_stress(index=i)
                pipe.checkpoint_data[pkey] = {"index": i + 1}

        mpi.comm.barrier()

        pipe.smooth_window_trajectory()

        mpi.comm.barrier()
        pkey = "extract windows"
        if not pipe.settings[pkey].get("skip", False):

            rdict: dict = pipe.checkpoint_data.get(pkey, {})
            begin: int = rdict.get("index", 0)

            for i in sorted(pipe.model.plt_files["by index"].keys())[begin:]:
                pipe.extract_windows(index=i)
                pipe.checkpoint_data[pkey] = {"index": i + 1}

        mpi.comm.barrier()
        pipe.refresh_model()

        mpi.comm.barrier()
        pkey = "analyze uniform data"

        rdict: dict = pipe.checkpoint_data.get(pkey, {})
        if pkey not in pipe.checkpoint_data:
            pipe.checkpoint_data[pkey] = {}
        begin: int = rdict.get("index", 0)

        for i in sorted(pipe.model.uni_files["by index"].keys())[begin:]:
            pipe.analyze_uniform_data(i)
            pipe.checkpoint_data[pkey]["index"] = i + 1

        mpi.comm.barrier()
        if mpi.root:
            print("DONE!")


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except Exception as exc:
        LOGGER.exception("", exc_info=exc)
        if mpi.procs > 1:
            mpi.comm.Abort()
