import json
from pathlib import Path

import h5py
import numpy as np


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

    def smooth_window_trajectory(self) -> None:

        if mpi.root:
            xmax = np.zeros(self.model.nfiles)
            time = np.zeros_like(xmax)

            for i, p in enumerate(sorted(self.model.plt_files["by_index"].keys())):

                self.model.load(file_number=p, file_type="plt")

                _path: str = self.model.plt_files["by index"][p].stem
                _path = _path.replace("plt_cnt", "analysis")
                _path = _path.replace("chk", "analysis")
                _path = _path.replace("part", "analysis")
                fn: Path = self.output_dir / _path

                with h5py.File(fn, "r") as f:
                    win_right = f["scalars"]["window right"][()]

                xmax[i] = win_right[0]
                time[i] = self.model.mesh.time

        g_xmax = np.zeros_like(xmax)
        g_time = np.zeros_like(time)

        mpi.comm.Allreduce(xmax, g_xmax)
        mpi.comm.Allreduce(time, g_time)

        coef = np.polyfit(g_time, g_xmax, 1)
        self.t0: float = g_time[0]
        self.x0: float = g_xmax[0]
        self.func = np.poly1d(coef)

    def reynolds_stress(self, index: int) -> None:

        self.model.load(file_index=index, file_type="plt")

        _path: str = Path(self.model.mesh.filename).stem
        _path = _path.replace("plt_cnt", "analysis")
        _path = _path.replace("chk", "analysis")
        _path = _path.replace("part", "analysis")
        fn: Path = self.output_dir / _path

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

        flam: str = "rpv1"
        try:
            self.model.mesh.data(flam)
        except:
            flam = "flam"
            self.model.mesh.data(flam)

        span, alp = self.model.slice_average(flam, axis=0)
        ccspan = 0.5 * (span[1:] + span[:-1])

        ccx = 0.5 * (x[1:] + x[:-1])

        mask = np.argwhere((0.0 < alp) & (alp < 1.0)).flatten()

        xmin, xmax = self.model.mesh.flame_window(ccx, s, mask)

        ymin, ymax = (self.model.mesh.ymin, self.model.mesh.ymax)
        zmin, zmax = (self.model.mesh.zmin, self.model.mesh.zmax)
        left = self.model.mesh.domain_bounds[:, 0]
        right = self.model.mesh.domain_bounds[:, 1]

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

        mpi.comm.barrier()


def main() -> None:
    pipe = Pipeline()
    pipe.load_settings(settings_path=Path("pipeline_settings.json"))

    for i in sorted(pipe.model.plt_files["by index"].keys()):
        pipe.reynolds_stress(index=i)

    pipe.smooth_window_trajectory()

    if mpi.root:
        print("DONE!")


if __name__ == "__main__":
    import sys

    sys.exit(main())
