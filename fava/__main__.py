import copy
import json
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from fava.model import FLASH
from fava.mesh.FLASH._flash import mpi


class Pipeline:

    # def __init__(self) -> None:
    #     self.load_settings()

    def load_settings(self, settings_path: Path) -> None:
        with settings_path.open("r") as f:
            self.settings: dict = json.load(f)

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

    def refresh_model(self) -> None:
        del self.model
        self.model = FLASH(self.data_dir)

    def reynolds_stress(self, index: int) -> None:

        self.model.load(file_index=index, file_type="plt")

        fn: Path = self.output_dir / self.convert_filename_type("plt", "anl").stem

        if mpi.root:
            print(fn, flush=True)
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
        # ymin, ymax = (self.model.mesh.ymin, self.model.mesh.ymax)
        # zmin, zmax = (self.model.mesh.zmin, self.model.mesh.zmax)
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

            fn: Path = self.output_dir / self.convert_filename_type("plt", "anl").stem

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

        fn: Path = self.output_dir / self.convert_filename_type("plt", "uni").stem

        if fn.is_file():
            return

        self.model.mesh.from_amr(subdomain_coords=subdomain_coords, fields=fields, filename=fn)

    def analyze_uniform_data(self, index: int) -> None:
        self.model.load(file_index=index, file_type="uni")

        success: bool = self._flam_or_rpv1()
        if not success:
            return

        fn: Path = self.output_dir / self.convert_filename_type("uni", "anl").stem

        akey: str = "fractal dimension"
        if not self.settings[akey].get("skip", False):
            contours: list[float] = self.settings[akey].get("contours", [0.5])
            results: dict = {}

            for contour in contours:
                retval: dict = self.model.fractal_dimension(self.flam, contour)
                results[f"{contour}"] = copy.deepcopy(retval)

            mpi.comm.barrier()
            if mpi.root:
                self.model.save_to_hdf5(data={akey: {"flame progress": results}}, filename=fn)
            mpi.comm.barrier()

        akey = "structure functions"
        if not self.settings[akey].get("skip", False):
            retval: dict = self.model.structure_functions()
            mpi.comm.barrier()
            self.model.save_to_hdf5(data={akey: retval}, filename=fn)
            mpi.comm.barrier()

        akey = "kinetic energy spectra"
        if not self.settings[akey].get("skip", False):
            retval: dict = self.model.kinetic_energy_spectra()
            mpi.comm.barrier()
            self.model.save_to_hdf5(data={akey: retval}, filename=fn)
            mpi.comm.barrier()


def main() -> None:
    pipe = Pipeline()
    pipe.load_settings(settings_path=Path("pipeline_settings.json"))

    if not pipe.settings["reynolds stress"].get("skip", False):
        for i in sorted(pipe.model.plt_files["by index"].keys()):
            pipe.reynolds_stress(index=i)

    mpi.comm.barrier()
    if not pipe.settings["smooth window trajectory"].get("skip", False):
        pipe.smooth_window_trajectory()

    mpi.comm.barrier()
    if not pipe.settings["extract windows"].get("skip", False):
        for i in sorted(pipe.model.plt_files["by index"].keys()):
            pipe.extract_windows(index=i)

    mpi.comm.barrier()
    pipe.refresh_model()

    mpi.comm.barrier()
    for i in sorted(pipe.model.uni_files["by index"].keys()):
        pipe.analyze_uniform_data()

    mpi.comm.barrier()
    if mpi.root:
        print("DONE!")


if __name__ == "__main__":
    import sys

    sys.exit(main())
