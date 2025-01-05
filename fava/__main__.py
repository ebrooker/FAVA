from pathlib import Path
from fava import Model, FLASH
from fava.util import mpi
import numpy as np
from numpy.typing import NDArray


# FIELDS: list[str] = ["flam"]

FIELDS: list[str] = ["dens", "velx", "vely", "velz", "flam", "divv", "igtm"]


def main_amr() -> None:
    p = Path("/Users/ezrabrooker/SimulationData/rtflame/3d/rtfD3x128d66t159g119c42p")
    m = FLASH(p)
    m.load(file_number=2200, file_type="plt")
    m.mesh.load_data(names=FIELDS)
    sd: NDArray = np.array([[432e5, 464e5], [-16e5, 16e5], [-16e5, 16e5]])

    m.mesh.from_amr(subdomain_coords=sd, fields=FIELDS)


def main_uni() -> None:
    p = Path("/Users/ezrabrooker/SimulationData/rtflame/3d/rtfD3x128d66t159g119c42p")
    m = FLASH(p)
    m.load(file_index=0, file_type="uni")
    m.mesh.load_data(names=FIELDS)

    if mpi.root:
        print(m.mesh.data("flam").min(), m.mesh.data("flam").max())

    # for c in [1e-4, 1e-2, 1e-1, 0.5, 0.9, 0.99]:
    #     d: float = m.fractal_dimension("flam", c)["average fractal dimension"]
    #     if mpi.root:
    #         print(f"\t{c} {d:.4f}", flush=True)

    kespec = m.kinetic_energy_spectra()


if __name__ == "__main__":
    # main_amr()

    main_uni()
