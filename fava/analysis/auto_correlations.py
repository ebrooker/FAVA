import numpy as np

from typing import Sequence, Tuple, Dict

from fava.model import Model

@Model.register_analysis(use_timer=True)
def eulerian_autocorrelation(self, nsamples: int, fields: Sequence[str], *args, **kwargs) -> Tuple[np.ndarray, Dict[str,np.ndarray]]:

    if "file_type" not in kwargs:
        kwargs["file_type"] = "plt"

    nfiles: int = self.nfiles(*args, **kwargs)
    time_seps: np.ndarray = np.empty(nfiles, dtype=float)
    results: Dict[str,np.ndarray] = {field: np.empty(nfiles, dtype=float) for field in fields}

    self.load(file_index=0, fields=fields, *args, **kwargs)

    if self.mesh is None:
        msg = "Eulerian autocorrelation requires an Eulerian mesh!"
        if self.particles is not None:
            msg += "Particles were loaded, possibly by mistake. Lagrangian autocorrelation uses particles!"
        raise RuntimeError(msg)

    lref_cells = 2**(self.mesh.lrefmax-1)

    dims = [nb*bl*lref_cells for nb,bl in zip(self.mesh.nCellsVec[:self.mesh.ndim], self.mesh.nBlksVec[:self.mesh.ndim])]

    points = np.empty((nsamples, self.mesh.ndim), dtype=float)
    dom_bnds = self.mesh.domain_bounds

    for nd in range(self.mesh.ndim):
        delta = (dom_bnds[nd,1] - dom_bnds[nd,0]) / float(dims[nd]+1)
        delta2 = 0.5e0 * delta
        ipnts = np.random.randint(low=0, high=dims[nd], size=nsamples)
        points[:,nd] = np.linspace(dom_bnds[nd,0] + delta2, dom_bnds[nd,1] - delta2, dims[nd])[ipnts]


    # Accumulate the later timestamp data
    for i in range(nfiles):
        try:
            self.load(file_index=i, fields=fields, *args, **kwargs)
        except Exception:
            print(f"Bad file: index={i}")
            continue
        time_seps[i] = self.mesh.time

        blk_list = self.mesh.get_list_of_blocks()

        if i == 0:
            init_data = {field: np.empty(nsamples, dtype=float) for field in fields}
            current_data = {field: np.empty(nsamples, dtype=float) for field in fields}

            for p in range(nsamples):
                point, blkID = self.mesh.get_coord_index(points[p,:], blk_list)
                vol_frac = self.mesh.get_cell_volume(blkID) / self.mesh.cell_volume_min

                for field in fields:
                    init_data[field][p] = self.mesh.get_point_data(blkID, point, field) * vol_frac

            for field in fields:
                current_data[field][:] = init_data[field][:]

            init_sum = {field: np.sqrt(np.sum(data**2)) for field, data in init_data.items()}

        else:
            for p in range(nsamples):
                point, blkID = self.mesh.get_coord_index(points[p,:], blk_list)
                vol_frac = self.mesh.get_cell_volume(blkID) / self.mesh.cell_volume_min
                for field in fields:
                    current_data[field][p] = self.mesh.get_point_data(blkID, point, field) * vol_frac

        for field in fields:
            results[field][i] += np.sum(init_data[field] * current_data[field]) / (init_sum[field] * np.sqrt(np.sum(current_data[field]**2)))


    return time_seps, results

@Model.register_analysis(use_timer=True)
def lagrangian_autocorrelation(self, nsamples: int, fields: Sequence[str], *args, **kwargs) -> Tuple[np.ndarray, Dict[str,np.ndarray]]:

    if "file_type" not in kwargs:
        kwargs["file_type"] = "prt"

    nfiles: int = self.nfiles(*args, **kwargs)

    time_seps: np.ndarray = np.empty(nfiles, dtype=float)
    results: Dict[str,np.ndarray] = {field: np.empty(nfiles, dtype=float) for field in fields}

    self.load(file_index=0, fields=fields, *args, **kwargs)

    if self.particles is None:
        msg = "Lagrangian autocorrelation requires Lagrangian Particles!"
        if self.mesh is not None:
            msg += "Only mesh was loaded, possibly by mistake. Eulerian autocorrelation uses a mesh!"
        raise RuntimeError(msg)


    for i in range(nfiles):
        self.load(file_index=i, fields=fields, *args, **kwargs)

        if i == 0:
            init_data = {field: np.copy(self.particles.data[field]) for field in fields}
            init_sum = {field: np.sqrt(np.sum(data**2)) for field, data in init_data.items()}

        time_seps[i] = self.particles.time

        for field in fields:
            results[field][i] += np.sum(init_data[field] * self.particles.data[field]) / (init_sum[field] * np.sqrt(np.sum(self.particles.data[field]**2)))


    return time_seps, results
