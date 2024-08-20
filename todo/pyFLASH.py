import sys

if sys.version_info[0] < 3:
    raise Exception(
        "[VERSION_ERROR] Python 2 is no longer supported, please use Python 3"
    )

import copy
import pathlib
import time
from math import log2

import FedTools
import h5py as h5
import histogram
import numpy as np
import scipy.optimize as sciopt
import yt

TIMERS = {}


def timer(func):
    def decorator(*args, **kwargs):
        tbeg = time.perf_counter()
        result = func(*args, **kwargs)
        tend = time.perf_counter()
        TIMERS[func.__name__] = tend - tbeg
        print(f"Timing: {func.__name__} --> {tend-tbeg:2.4f}")
        return result

    return decorator


class Block:
    def __init__(self, dims):
        self.dims = np.copy(dims)
        self.bounds = None
        self.dims = None
        self.lref = None
        self.size = None


def super_gaussian(x, amp, x0, sigma):
    return amp * np.exp(-2 * ((x - x0) / sigma) ** 10)


def curve_fit(func, x, y, p0):
    return sciopt.curve_fit(func, x, y, method="lm", p0=p0)


class pyFLASH:

    def __init__(self, whoami="FLASH_API"):

        self._whoami = whoami
        print("[{0}] Constructing data analysis object".format(self._whoami))

        self.ds_open = False
        self._get_inds = False
        self._verbose = False

    def __del__(self):
        if self.ds_open:
            self.FLASH_close()
        print("[{0}] Destroyed data analysis object!".format(self._whoami))

    def _super_gaussian(x, amp, x0, sigma):
        return amp * np.exp(-2 * ((x - x0) / sigma) ** 10)

    @timer
    def FLASH_load(self, path):

        print("[{0}] Opening dataset: {1}".format(self._whoami, path))
        self.path = path
        _path = _path.with_name("plt_cnt", "analysis")
        _path = _path.with_name("chk", "analysis")
        _path = _path.with_name("part", "part_analysis")
        self._analysis_base = _path
        self.ds = yt.load(self.path)
        self.file = self.path.split("/")[-1]
        self.ds_open = True

    @timer
    def FLASH_close(self):
        if self.ds_open:
            print("[{0}] Closing dataset: {1}".format(self._whoami, self.path))
        else:
            print("[{0}] No dataset is open. Strange...".format(self._whoami))

        self.ds_open = False
        self.ds.close()

    @timer
    def FLASH_props(self):

        self.basename = self.ds.basename
        self.geometry = self.ds.geometry
        self.ndim = self.ds.dimensionality
        self.minDim = self.ds.domain_dimensions
        self.maxDim = np.copy(self.minDim)
        self.lrefMax = self.ds.index.max_level
        self.maxDim[: self.ndim] *= 2**self.lrefMax
        self.maxcells = self.maxDim.prod()
        self.mincells = self.minDim.prod()
        self.right = self.ds.domain_right_edge.d
        self.left = self.ds.domain_left_edge.d
        self.domain = self.right - self.left
        self.maxDelta = self.domain / self.minDim
        self.minDelta = self.domain / self.maxDim
        self.cellVol = np.product(self.minDelta)
        self.time = self.ds.current_time.d

        print("Basename: ", self.basename)

    @timer
    def getWindow(self):

        self.no_win = True
        self.win_left = np.copy(self.left)
        self.win_right = np.copy(self.right)

        dx = np.min(self.domain[0 : self.ndim])

        self.win_right[0 : self.ndim] = dx
        self.win_domain = self.win_right - self.win_left
        self.win_dims = (self.win_domain / self.maxDelta).astype(np.int64)
        self.no_win = False

    @timer
    def getCube(self):
        self.cube = self.ds.covering_grid(
            self.lrefMax, left_edge=self.win_left, dims=self.win_dims
        )

    @timer
    def getIndices(self, win_var, win_var_scale):

        self.win_var = win_var
        self.win_var_scale = win_var_scale
        _win_var = self.win_var_scale * self.cube[self.win_var].d

        self.indices = {}
        self.indices["Domain"] = _win_var < np.inf

        if not self._get_inds:
            del [_win_var]
            return

        del [_win_var]

    @timer
    def write_data(self, handle, data):
        # Iterate through the items in the data dictionary
        for key, values in data.items():

            # If the values for this key is another dictionary, we need to go one group level down
            if isinstance(values, dict):
                # Create a group handle and pass it and the sub-dict to write_data()
                group = handle.create_group(key)
                self.write_data(group, values)

            # No more sub-groups we can write the dataset now.
            else:
                try:
                    _ = values.copy()
                    handle.create_dataset(key, data=values.copy())
                except:
                    _ = copy.copy(values)
                    handle.create_dataset(key, data=copy.copy(values))

    @timer
    def save_data(self, data: dict):
        # Save the data to the analysis file, append mode if it already exists
        mode = "a" if self._baseresults.is_file() else "w"
        with h5.File(str(self._baseresults), mode) as f:
            # Start recursive writing of the data dictionary
            self.write_data(f, data)

    @timer
    def getPDF1d(self, fields, bounds):
        wstr = ""
        mass = self.cube["dens"].d * self.cellVol

        _data = {self.turb: {}}

        for fi, fstr in enumerate(fields):
            fl_str = fstr

            _data[self.turb][fstr] = {}

            if fstr == "mach":
                try:
                    data = np.sqrt(
                        (self.cube["velx"].d - self.cube["velx"].d.mean()) ** 2
                        + self.cube["vely"].d ** 2
                        + self.cube["velz"].d ** 2
                    ) / np.sqrt(
                        self.cube["gamc"].d * self.cube["pres"].d / self.cube["dens"].d
                    )
                except:
                    data = np.sqrt(
                        (self.cube["velx"].d - self.cube["velx"].d.mean()) ** 2
                        + self.cube["vely"].d ** 2
                        + self.cube["velz"].d ** 2
                    ) / np.sqrt(self.cube["pres"].d / self.cube["dens"].d)

            elif "_dot_" in fstr:
                Astr, Bstr = fstr.split("_dot_")
                adata = np.copy(self.cube[Astr[0:4]].d)
                bdata = np.copy(self.cube[Bstr[0:4]].d)

                if "velx" in Astr:
                    adata -= adata.mean()
                if "velx" in Bstr:
                    bdata -= bdata.mean()
                data = adata * bdata

            else:
                data = np.copy(self.cube[fstr[:4]].d)

            if "pres" in fstr:
                data /= np.mean(data)

            if fstr in ("velx", "velxFuelWeighted", "velxFlameWeighted"):
                data -= np.mean(data)

            if "FlameWeighted" in fstr:
                try:
                    massf = mass * self.cube["rpv1"].d
                except:
                    massf = mass * self.cube["flam"].d
            elif "FuelWeighted" in fstr:
                try:
                    massf = mass * (1.0 - self.cube["rpv1"].d)

                except:
                    massf = mass * (1.0 - self.cube["flam"].d)
            else:
                massf = np.copy(mass)

            if "igtm" in fstr:
                data = np.log10(data)
            if "dens" in fstr:
                data /= np.mean(data)
                data = np.log10(data)

            for ii, istr in enumerate(list(self.indices.keys())):
                _data[self.turb][fstr][istr] = {}
                H, bin_edges = histogram.compute_weighted(
                    data[self.indices[istr]].ravel(),
                    massf[self.indices[istr]].ravel(),
                    200,
                    histogram.BIN_SPACING.linear,
                    bounds[fstr][0],
                    bounds[fstr][1],
                )
                H /= mass.sum()

                _data[self.turb][fstr][istr]["pdf"] = np.copy(H)
                _data[self.turb][fstr][istr]["bins"] = np.copy(
                    0.5 * (bin_edges[1:] + bin_edges[:-1])
                )

                # fn = "{0}/{1}_{2}_{3}_{4}_{5}_1d_profile".format(
                #     self.bdir, self.bname, self.turb, istr, fl_str, wstr
                # )

                # self.save_data({self.turb: {istr: {fl_str: {}}}})

                # with open(fn, "w") as fout:
                #     fout.write("bin             {0}\n".format(fstr))

                #     bw = 0.5 * (bin_edges[:-1] - bin_edges[1:])
                #     fout.write(
                #         "{0} {1}\n".format(bin_edges[0] - bw[0], 0.0)
                #     )  # Make a zero bin on LHS

                #     for k, bin_ in enumerate(bin_edges[:-1] + bw):
                #         fout.write("{0} {1}\n".format(bin_, H[k]))

                #     fout.write(
                #         "{0} {1}\n".format(bin_edges[-1] + bw[-1], 0.0)
                #     )  # Make a zero bin on RHS
                H = None
                del [H]
            # _data[self.turb][fstr]["massf"] = massf.sum()

            data = None
            del [data]
        _data[self.turb]["mass sum"] = mass.sum()
        mass = None
        del [mass]
        self.save_data({"1D PDFs": _data})

    @timer
    def save_scalars(self):
        _data = {"time": self.ds.current_time.d}
        self.save_data({"scalars": _data})
        _data = None
        del _data

    @timer
    def velocity_rms(self):
        """Returns the root mean square (RMS) of the field provided (pass the parameter <particles=True> to use particle data instead of mesh data)"""
        rms = {"time": self.ds.current_time.d}
        _data = {self.turb: {}}
        for field in ("velx", "vely", "velz", "vtot", "Mach"):

            if field == "velx":
                data = self.cube[field].d
                data -= data.mean()

            elif field in ("vely", "velz"):
                data = self.cube[field].d

            elif field == "vtot":
                data = np.sqrt(
                    (self.cube["velx"].d - self.cube["velx"].d.mean()) ** 2
                    + self.cube["vely"].d ** 2
                    + self.cube["velz"].d ** 2
                )

            elif field == "Mach":
                try:
                    data = np.sqrt(
                        (self.cube["velx"].d - self.cube["velx"].d.mean()) ** 2
                        + self.cube["vely"].d ** 2
                        + self.cube["velz"].d ** 2
                    ) / np.sqrt(
                        self.cube["gamc"].d * self.cube["pres"].d / self.cube["dens"].d
                    )
                except:
                    data = np.sqrt(
                        (self.cube["velx"].d - self.cube["velx"].d.mean()) ** 2
                        + self.cube["vely"].d ** 2
                        + self.cube["velz"].d ** 2
                    ) / np.sqrt(self.cube["pres"].d / self.cube["dens"].d)

            _data[self.turb][field] = np.sqrt(np.mean(data**2))
            # rms[field] = np.sqrt(np.mean(data**2))

        self.save_data({"RMS": _data})
        rms = None
        del rms
        # fn = f"{self.bdir}/{self.bname}_{self.turb}_rms"
        # with open(fn, "w") as f:
        #     rms_keys = list(rms.keys())
        #     f.write("".join(colhead + "   " for colhead in rms_keys) + "\n")

        #     fstr = ""
        #     for key in rms_keys:
        #         fstr += f"{rms[key]}    "
        #     f.write(fstr + "\n")

    @timer
    def getTurbVelStats(self, fields, bounds):

        wstr = ""
        mass = self.cube["dens"].d * self.cellVol

        _data = {self.turb: {}}

        for fi, fstr in enumerate(fields):
            fl_str = fstr + "-fluc"
            data = self.cube[fstr].d

            if fi == 0:
                data -= data.mean()

            _data[self.turb][fl_str] = {}

            for ii, istr in enumerate(list(self.indices.keys())):
                _data[fl_str][istr] = {}
                H, bin_edges = histogram.compute_weighted(
                    data[self.indices[istr]].flatten(),
                    mass[self.indices[istr]].flatten(),
                    200,
                    histogram.BIN_SPACING.linear,
                    *bounds[fstr],
                )

                H /= mass.sum()

                _data[self.turb][fl_str][istr]["pdf"] = np.copy(H)
                _data[self.turb][fl_str][istr]["bins"] = np.copy(
                    0.5(bin_edges[1:] + bin_edges[:-1])
                )

                # fn = "{0}/{1}_{2}_{3}_{4}_{5}_1d_profile".format(
                #     self.bdir, self.bname, self.turb, istr, fl_str, wstr
                # )

                # with open(fn, "w") as fout:
                #     fout.write("bin             {0}\n".format(fstr))
                #     for k, bin_ in enumerate(0.5 * (bin_edges[:-1] + bin_edges[1:])):
                #         fout.write("{0} {1}\n".format(bin_, H[k]))
                H = None
                del [H]
            data = None
            del [data]
        mass = None
        del [mass]
        self.save_data({"turbulent velocity statistics": _data})

    @timer
    def getRadialReynoldStress(self, raxis=0, coord_convert=None, find_window=True):
        """
        print("Computing Reynolds Stresses - New Way")
        rho  = self.cube["dens"].d
        rho0 = np.mean(rho)**(-1)
        x    = self.cube.fcoords[:,0].reshape(self.maxDim)[:,0,0]
        vel  = np.zeros((3,*self.maxDim))

        vel[0,...] = self.cube["velx"].d
        if self.ndim > 1: vel[1,...] = self.cube["vely"].d
        if self.ndim > 2: vel[2,...] = self.cube["velz"].d

        # Get mean velocity in x, y, and z each
        vmean = [np.mean(vel[i,...]) for i in range(3)]

        # Store the results of Reynold Stress matrix in a dictionary for easy referencing w/ labels
        xyz = "xyz"
        RS  = {}
        for i in range(self.ndim):
            for j in range(i,self.ndim):
                RS["R{0}{1}".format(xyz[i],xyz[j])] = np.ones(x.size) * rho0


        for i in range(self.ndim):
            for j in range(i,self.ndim):
                RSstr = "R{0}{1}".format(xyz[i],xyz[j])

                # List comprehension through the radial bins
                RS[RSstr] *= np.array([np.mean(np.abs(rho[k,:,:]*vel[i,k,:,:] * (vel[j,k,:,:]-vmean[j]))) for k in range(x.size)])


        del[vel]
        del[rho]

        fn = "{0}/{1}_radialReynoldStress".format(self.bdir, self.bname)

        with open(fn,"w") as fout:
            fout.write("radius ")
            [fout.write("      {0}".format(k)) for k in RS.keys()]
            fout.write(f"\n")

            for i in range(x.size):
                fout.write("{0} ".format(x[i]))
                [fout.write("{0} ".format(RS[k][i])) for k in RS.keys()]
                fout.write("\n")


        if find_window:
         # Get half burned indices
         try:
            rpv1 = self.cube["rpv1"].d
         except:
             rpv1 = self.cube["flam"].d
         idx50 = np.array(np.where( (rpv1 >= 0.45) & (rpv1 <= 0.55) ))
         x50 = x[idx50[0,:]]

         print(f"xmin = {x50.min()/1e5} -- xavg = {x50.mean()/1e5} -- xmax = {x50.max()/1e5}")
         x50m = np.mean(x50)

         idx50 = np.where( ( (x50m-64e5) <= x ) & ( x <= (x50m+64e5) ) )[0]
         x50 = x[idx50]/1e5
         x50m = np.mean(x50)

         print(f"xmin = {x50.min()} -- xavg = {x50m} -- xmax = {x50.max()}")

         RSxx50 = RS["Rxx"][idx50]/1e14
         opt_xx, _ = curve_fit( super_gaussian, x50-np.min(x50), RSxx50, (np.max(RSxx50), x50[np.argmax(RSxx50)], np.std(RSxx50)) )

         print(f"Rxx {opt_xx}")


         RSyyzz50 = RS["Ryy"][idx50]/1e14 + RS["Rzz"][idx50]/1e14
         opt_yyzz, _ = curve_fit( super_gaussian, x50-np.min(x50), RSyyzz50, (np.max(RSyyzz50), x50[np.argmax(RSyyzz50)], np.std(RSyyzz50)) )

         print(f"Ryy+zz {opt_yyzz}")

         fit = opt_xx[1] + opt_yyzz[1]
         fit *= 0.5

         fitmax = np.argmax(fit)

         print(fit)
         print(f"fitmax is {fitmax}")

         self.win_xmin = fit*1e5 - 16e5
         self.win_xmax = fit*1e5 + 16e5

        fn = "{0}/{1}_radialReynoldStress".format(self.bdir, self.bname)
        """
        import fava.model

        model = fava.model.Model("./")
        model.load(self.path)
        x, RS = model.mesh.reynolds_stress()
        self.win_xmin, self.win_xmax = model.mesh.flame_window(x, RS)

        x = 0.5 * (x[1:] + x[:-1])
        fn = "{0}/{1}_radialReynoldStress".format(self.bdir, self.bname)
        with open(fn, "w") as fout:
            fout.write("radius ")
            [fout.write("      {0}".format(k)) for k in RS.keys()]
            fout.write(f"\n")

            for i in range(x.size):
                fout.write("{0} ".format(x[i]))
                [fout.write("{0} ".format(RS[k][i])) for k in RS.keys()]
                fout.write("\n")

        _data = {"tensor": copy.deepcopy(RS), "radius": np.copy(x)}
        self.save_data({"reynolds stresses": _data})
        RS = None
        del [RS]
        x = None
        del [x]
        # del [rho0]

    @timer
    def set_velocity_data(self):
        lrefmax = -1
        with h5.File("decomp_hdf5_chk_0001", "r+") as ds:
            intscalars = {
                tpl[0].strip().decode("UTF-8"): tpl[1]
                for tpl in ds["integer scalars"][()]
            }
            intrunpars = {
                tpl[0].strip().decode("UTF-8"): tpl[1]
                for tpl in ds["integer runtime parameters"][()]
            }
            realrunpars = {
                tpl[0].strip().decode("UTF-8"): tpl[1]
                for tpl in ds["real runtime parameters"][()]
            }

            nx = 16  # intrunpars["nblockx"]
            ny = 16  # intrunpars["nblocky"]
            nz = 16  # intrunpars["nblockz"]
            nblks = (
                intscalars["total blocks"]
                if "total blocks" in intscalars
                else intscalars["globalnumblocks"]
            )
            dims = np.array([nx, ny, nz])
            blocks = [Block(dims) for _ in range(nblks)]

            for i, (bounds, lref, size) in enumerate(
                zip(
                    ds["bounding box"][()], ds["refine level"][()], ds["block size"][()]
                )
            ):
                blocks[i].bounds = np.copy(bounds)
                blocks[i].lref = lref
                blocks[i].size = np.copy(size)

                lrefmax = max(lrefmax, lref)

            velx = self.cube["velx"].d
            vely = self.cube["vely"].d
            velz = self.cube["velz"].d
            mx, my, mz = velx.shape

            xmin, xmax = realrunpars["xmin"], realrunpars["xmax"]
            ymin, ymax = realrunpars["ymin"], realrunpars["ymax"]
            zmin, zmax = realrunpars["zmin"], realrunpars["zmax"]

            MBX, MBY, MBZ = mx // nx, my // ny, mz // nz

            dx = 0.5 * ((xmax - xmin) / mx)
            dy = 0.5 * ((ymax - ymin) / my)
            dz = 0.5 * ((zmax - zmin) / mz)

            x = np.linspace(xmin + dx, xmax - dx, mx)
            y = np.linspace(ymin + dy, ymax - dy, my)
            z = np.linspace(zmin + dz, zmax - dz, mz)

            for b, blk in enumerate(blocks):

                if blk.lref == lrefmax:
                    i = np.where(((blk.bounds[0, 0] < x) & (x < blk.bounds[0, 1])))[0]
                    j = np.where(((blk.bounds[1, 0] < y) & (y < blk.bounds[1, 1])))[0]
                    k = np.where(((blk.bounds[2, 0] < z) & (z < blk.bounds[2, 1])))[0]

                    ilo = i.min()
                    ihi = i.max() + 1
                    jlo = j.min()
                    jhi = j.max() + 1
                    klo = k.min()
                    khi = k.max() + 1
                    ds["velx"][b, ...] = velx[ilo:ihi, jlo:jhi, klo:khi]
                    ds["vely"][b, ...] = vely[ilo:ihi, jlo:jhi, klo:khi]
                    ds["velz"][b, ...] = velz[ilo:ihi, jlo:jhi, klo:khi]

    @timer
    def getKESpectra(self):

        if self._verbose:
            print("{0}Computing kinetic energy spectra".format(" " * 4))

        nx, ny, nz = self.win_dims

        din = np.ones((self.ndim, nx, ny, nz)) * np.sqrt(self.cube["dens"].d)

        vels = ["velx", "vely", "velz"]

        for i, v in enumerate(vels[0 : self.ndim]):
            din[i, ...] *= self.cube[v].d

        ke = {}

        if True:
            # for istr in list(self.indices.keys()):

            # if True:
            #     igtm = np.log10(self.cube["igtm"].d[self.indices[istr]])
            #     divv = self.cube["divv"].d[self.indices[istr]]
            #     try:
            #         rpv1 = self.cube["rpv1"].d[self.indices[istr]]
            #     except:
            #         rpv1 = self.cube["flam"].d[self.indices[istr]]
            #     print(f"STATS --> {igtm.mean()}, {divv.mean()}, {rpv1.mean()}")
            #     with open(f"{self.bdir}/{self.bname}_{self.turb}_{istr}_igtm_divv_rpv1.dat", "w") as f:
            #         f.write("rpv1    igtm    divv\n")
            #         for ii,jj,kk in zip(rpv1.flatten(), igtm.flatten(), divv.flatten()):
            #             f.write(f"{ii}    {jj}    {kk}\n")

            #   if self._verbose:
            #       print("{0} - {1}".format(" "*12,istr))
            dcopy = din.copy()
            #   dcopy[:,~self.indices[istr]] = 0.e0
            ke = FedTools.get_spectrum(np.squeeze(dcopy), ncmp=self.ndim)

            k = np.copy(ke["k"])
            E = np.copy(ke["P_tot"])
            intg = np.zeros_like(k)
            intguw = np.zeros_like(intg)
            for i in range(intg.size):
                if k[i] > 0:
                    dk = k[i] - k[i - 1]
                    intguw[i] = intguw[i - 1] + E[i] * dk
                    intg[i] = intg[i - 1] + (k[i] ** -1) * E[i] * dk

            ke["KEintg"] = intguw
            ke["length"] = 0.5e0 * np.pi * intg / intguw

            for i, v in enumerate(vels[0 : self.ndim]):
                din[i, ...] = self.cube[v].d

            din[0, ...] -= np.mean(din[0, ...])

            U = np.sqrt(
                (
                    (self.cube["velx"].d - self.cube["velx"].d.mean()) ** 2
                    + self.cube["vely"].d ** 2
                    + self.cube["velz"].d ** 2
                )
            )

            ke["eddy_tau"] = ke["length"] * self.win_domain[0] / U.mean()
            # print(f"RMS Velocity {U.mean()}")

            # intg = np.zeros_like(k)
            # intguw = np.zeros_like(intg)
            # for i in range(1, intg.size):
            #     dk = k[i] - k[i - 1]
            #     intg[i] = intg[i - 1] + 2.0 * E[i] * dk

            # edsp = (E[1:] / (k[1:] ** (-5.0 / 3.0))) ** (3.0 / 2.0)
            # intg[1:] = (U.mean()) ** (3.0 / 2.0) / edsp
            # KEU = (
            #     0.5
            #     * np.sqrt(
            #         (
            #             (
            #                 (self.cube["velx"].d - self.cube["velx"].d.mean()) ** 2
            #                 + self.cube["vely"].d ** 2
            #                 + self.cube["velz"].d ** 2
            #             )
            #         )
            #         / nn
            #     )
            #     ** 2
            #     * self.cube["dens"].d
            #     * self.cellVol
            # )
            # ke["Etotal"] = np.zeros_like(ke["k"]) + KEU.sum()
            # ke["Emean"] = np.zeros_like(ke["k"]) + KEU.mean()
            # ke["integral_l"] = np.copy(intg)
            # ke["integral_t"] = ke["integral_l"] * self.win_domain[0] / U.mean()
            # ke["integral_k"] = 1.0 / ke["integral_l"]

            self.save_data({"kinetic energy spectra": ke})

            # kefile = "{0}/{1}_{2}_{3}_keSpectra".format(
            #     self.bdir, self.bname, self.turb, istr
            # )
            # with open(kefile, "w") as f:
            #     ke_keys = list(ke.keys())
            #     f.write("".join(colhead + "   " for colhead in ke_keys) + "\n")

            #     for i in range(ke[ke_keys[0]].size):
            #         fstr = (
            #             "".join("{0}   ".format(v[i]) for v in ke.values()) + "\n"
            #         )
            #         f.write(fstr)

        del [din]

    @timer
    def structure_functions(self):
        min_sep = 6.25e3
        max_sep = 30.0e5
        is_log_scale = True

        nseps = 100
        npts = 100000

        # Compute separations
        if is_log_scale:
            separations = np.geomspace(min_sep, max_sep, nseps)
        else:
            separations = np.linspace(min_sep, max_sep, nseps)

        xhi, yhi, zhi = self.win_right
        xlo, ylo, zlo = self.win_left

        SFs = {"transverse": {}, "longitudinal": {}, "separations": separations.copy()}
        for order in range(2, 12, 2):
            print(f"\n*************************\nWorking order {order}")
            pt_coords = np.empty((nseps, 2, npts, self.ndim))
            vel_comps = np.empty((nseps, npts, self.ndim))

            # -------------------------------------------
            # Generate local points
            # print("generate local points")
            pt_rand = np.empty((2, npts, self.ndim))
            for i, sep in enumerate(separations):
                # print(i,sep)
                pt_rand[0, :, :] = (
                    np.random.random((npts, self.ndim)) * self.win_domain
                    + self.win_left
                )
                # -------------------------------------------

                # -------------------------------------------
                # Generate point pairs
                # print("generate point pairs")
                phi = np.random.random(npts) * 2.0 * np.pi
                theta = np.arccos(2.0 * np.random.random(npts) - 1.0)

                newx = pt_rand[0, :, 0] + sep * np.sin(theta) * np.cos(phi)
                newy = pt_rand[0, :, 1] + sep * np.sin(theta) * np.sin(phi)
                newz = pt_rand[0, :, 2] + sep * np.cos(theta)

                # Bring y points within bounds (works if periodic)
                # print("bringing y within upper bounds")

                while np.any(newx > xhi):
                    k = np.where((newx > xhi))[0]
                    newx[k] += xlo - xhi

                # print("bringing x within lower bounds")
                while np.any(newx < xlo):
                    k = np.where((newx < xlo))[0]
                    newx[k] += xhi - xlo

                while np.any(newy > yhi):
                    k = np.where((newy > yhi))[0]
                    newy[k] += ylo - yhi

                # print("bringing y within lower bounds")
                while np.any(newy < ylo):
                    k = np.where((newy < ylo))[0]
                    newy[k] += yhi - ylo

                # Bring z points within bounds (works if periodic)
                # print("bringing z within upper bounds")
                while np.any(newz > zhi):
                    k = np.where((newz > zhi))[0]
                    newz[k] += zlo - zhi

                # print("bringing z within lower bounds")
                while np.any(newz < zlo):
                    k = np.where((newz < zlo))[0]
                    newz[k] += zhi - zlo

                pt_rand[1, :, 0] = newx
                pt_rand[1, :, 1] = newy
                pt_rand[1, :, 2] = newz
                # -------------------------------------------

                # -------------------------------------------
                # get velocity computations
                # print("computing cell size")
                cell_size = self.win_domain / self.win_dims
                pt1 = np.empty((pt_rand.shape[1:]), dtype=int)
                pt2 = np.empty((pt_rand.shape[1:]), dtype=int)
                for j in range(self.ndim):
                    pt1[:, j] = np.floor(
                        (pt_rand[0, :, j] - self.win_left[j]) / cell_size[j]
                    ).astype(int)
                    pt2[:, j] = np.floor(
                        (pt_rand[1, :, j] - self.win_left[j]) / cell_size[j]
                    ).astype(int)

                # Get velocity differences
                vel_comps[i, :, 0] = (
                    self.cube["velx"].d[pt2[:, 0], pt2[:, 1], pt2[:, 2]]
                    - self.cube["velx"].d[pt1[:, 0], pt1[:, 1], pt1[:, 2]]
                )
                vel_comps[i, :, 1] = (
                    self.cube["vely"].d[pt2[:, 0], pt2[:, 1], pt2[:, 2]]
                    - self.cube["vely"].d[pt1[:, 0], pt1[:, 1], pt1[:, 2]]
                )
                vel_comps[i, :, 2] = (
                    self.cube["velz"].d[pt2[:, 0], pt2[:, 1], pt2[:, 2]]
                    - self.cube["velz"].d[pt1[:, 0], pt1[:, 1], pt1[:, 2]]
                )

                pt_coords[i, 0, :, :] = pt_rand[0, :, :]
                pt_coords[i, 1, :, :] = pt_rand[1, :, :]

            # Get separation vector
            # print("creating separation vector")
            sep_vec = pt_coords[:, 1, :, :] - pt_coords[:, 0, :, :]

            # Normalize the vector into a unit vector
            # print("creating unit vector")
            rhat = np.empty_like(sep_vec)
            for i in range(self.ndim):
                rhat[:, :, i] = sep_vec[:, :, i] / np.sqrt(np.sum(sep_vec**2, axis=2))

            # Compute lognitudinal SFs
            # print("computing longitudinal SFs")
            long_comp = np.sum(vel_comps * rhat, axis=2)

            long_sfs = np.sum(long_comp**order, axis=1) / npts

            long_vel_diff = np.empty_like(sep_vec)
            for i in range(self.ndim):
                long_vel_diff[:, :, i] = long_comp * rhat[:, :, i]

            trans_comp = np.sqrt(np.sum((vel_comps - long_vel_diff) ** 2, axis=2))
            trans_sfs = np.sum(trans_comp**order, axis=1) / npts

            for i in range(nseps):
                for j in range(npts):
                    long_comp[i, j] = np.sum(vel_comps[i, j, :] * rhat[i, j, :])

            # SFs["transverse"]["separations"] = np.copy(separations)
            # SFs["longitudinal"]["separations"] = np.copy(separations)
            SFs["transverse"][f"order_{order}"] = np.copy(trans_sfs)
            SFs["longitudinal"][f"order_{order}"] = np.copy(long_sfs)

        self.save_data({"structure functions": SFs})
        # # print("writing to file")
        # for dkey in list(SFs.keys()):
        #     sffile = "{0}/{1}_{2}_{3}_structfunc".format(
        #         self.bdir, self.bname, self.turb, dkey
        #     )
        #     with open(sffile, "w") as f:
        #         sf_keys = list(SFs[dkey].keys())
        #         f.write("".join(colhead + "   " for colhead in sf_keys) + "\n")

        #         for i in range(SFs[dkey][sf_keys[0]].size):
        #             fstr = (
        #                 "".join("{0}   ".format(v[i]) for v in SFs[dkey].values())
        #                 + "\n"
        #             )
        #             f.write(fstr)

        SFs = None
        del SFs

    @timer
    def multi_fractal_dimension(self, field: str):
        qmin: int = -10
        qmax: int = 10
        falpha_size: int = qmax - qmin + 1
        qspan = np.linspace(qmin, qmax, falpha_size)
        try:
            data = self.cube[field].d
        except:
            if field == "rpv1":
                data = self.cube["flam"].d
            else:
                return
        height, width, depth = self.win_dims

        have_zeros = np.any(data == 0)

        correction = data.sum()
        if have_zeros:
            correction += float(height * width * depth)
            data[...] += 1.0
        data /= correction

        lowest_level = 0

        _data = {"falpha size": falpha_size, "q": qspan.copy(), "levels": {}}
        for level in range(lowest_level, int(log2(height))):
            print(f"\nLevel: {level}\n")

            bdim: int = int(2**level)
            bdim_k = bdim
            if self.ndim < 2:
                bdim_k = 1

            npts: int = int(height * width * depth / (bdim * bdim * bdim_k))
            falpha: np.ndarray = np.empty((falpha_size, 2))

            probs: np.ndarray = np.empty(npts)
            pidx = 0
            for i in range(0, height, bdim):
                for j in range(0, width, bdim):
                    for k in range(0, depth, bdim_k):

                        bsum: float = 0
                        for bx in range(bdim):
                            for by in range(bdim):
                                for bz in range(bdim_k):
                                    bsum += data[bx + i, by + j, bz + k]
                        probs[pidx] = bsum
                        # probs[pidx] = np.sum(data[i:i+bdim,j:j+bdim,k:k+bdim_k])
                        pidx += 1

            for q in qspan:
                pqsum = np.sum(probs**q)
                log2_size = log2(float(bdim / height))
                norm = (probs**q) / pqsum
                falpha[int(q - qmin)][1] = np.sum(norm * np.log2(norm)) / log2_size
                falpha[int(q + qmin)][0] = np.sum(norm * np.log2(probs)) / log2_size
            print(falpha, falpha_size, 2)

            _data["levels"][f"{level}"] = {
                "falpha": falpha[:, 1],
                "alpha": falpha[:, 0],
            }
            self.save_data({"multifractal dimensions": _data})

            # file_exists = pathlib.Path("multi_fractal_dimensions.dat").is_file()
            # with open("multi_fractal_dimensions.dat", "a") as f:
            #     if not file_exists:
            #         f.write(f"level    time    falpha_size    alpha    falpha\n")
            #     f.write(f"{self.time}    {falpha_size}\n")
            #     for y in falpha[:, 0]:
            #         f.write(f"{y}    ")
            #     f.write("\n")
            #     for y in falpha[:, 1]:
            #         f.write(f"{y}    ")
            #     f.write("\n")

    @timer
    def fractal_dimension(self, field: str, contour: float):
        try:
            data = self.cube[field].d
        except:
            data = self.cube["flam"].d
        height, width, depth = self.win_dims

        if contour is None:
            _contour = data.mean()
        else:
            _contour = contour

        edata = np.zeros_like(data)
        edata[np.where(data == 1)[0]] = 1

        depth_start = 0
        if depth != 1:
            depth_start += 1
        else:
            depth += 1

        _data = {"contour": _contour}

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                for k in range(depth_start, depth - 1):
                    val = data[i, j, k]
                    if val < _contour:
                        hidx = int((_contour - val))

                        if data[i + 1, j, k] > _contour:
                            if int(hidx / (data[i + 1, j, k] - val)) == 0:
                                edata[i, j, k] = 1
                            else:
                                edata[i + 1, j, k] = 1

                        if data[i, j + 1, k] > _contour:
                            if int(hidx / (data[i, j + 1, k] - val)) == 0:
                                edata[i, j, k] = 1
                            else:
                                edata[i, j + 1, k] = 1

                        if data[i, j - 1, k] > _contour:
                            if int(hidx / (data[i, j - 1, k] - val)) == 0:
                                edata[i, j, k] = 1
                            else:
                                edata[i, j - 1, k] = 1

                        if data[i - 1, j, k] > _contour:
                            if int(hidx / (data[i - 1, j, k] - val)) == 0:
                                edata[i, j, k] = 1
                            else:
                                edata[i - 1, j, k] = 1

                        if data[i, j, k + 1] > _contour:
                            if int(hidx / (data[i, j, k + 1] - val)) == 0:
                                edata[i, j, k] = 1
                            else:
                                edata[i, j, k + 1] = 1

                        if data[i, j, k - 1] > _contour:
                            if int(hidx / (data[i, j, k - 1] - val)) == 0:
                                edata[i, j, k] = 1
                            else:
                                edata[i, j, k - 1] = 1

        lowest_level: int = 0
        largest_dim: int = min(height, width)
        if depth > 1:
            largest_dim = min(largest_dim, depth)

        flength: int = int(log2(largest_dim) - lowest_level + 1)

        result = np.zeros((flength, 2))

        for level in range(lowest_level, flength + lowest_level):
            bdim = bdim_k = int(2**level)
            if self.ndim < 2:
                bdim_k = 1

            nfilled: int = 0
            for i in range(0, height, bdim):
                for j in range(0, width, bdim):
                    for k in range(0, depth, bdim_k):
                        bsum = 0
                        for bx in range(bdim):
                            for by in range(bdim):
                                for bz in range(bdim_k):
                                    bsum += edata[bx + i, by + j, bz + k]
                        if bsum > 0:
                            nfilled += 1

            result[level, 0] = flength - level - 1
            result[level, 1] = np.log2(nfilled)
        print(result)
        filled_boxes = 2 ** result[:, 1]
        cum_frac_dim = np.sum(np.log2(filled_boxes[:-1] / filled_boxes[1:]))
        avg_frac_dim = cum_frac_dim / (filled_boxes.size - 1)

        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        rval = np.sum((result[:, 0] - mean[0]) * (result[:, 1] - mean[1])) / (
            np.product(std) * result.shape[0]
        )
        slope = rval * std[1] / std[0]
        regress = np.array([slope, rval**2, mean[1] - slope * mean[0]])

        _data["average fractal dimension"] = avg_frac_dim
        _data["slope"] = regress[0]
        _data["R2"] = regress[1]
        _data["curve"] = regress[2]

        self.save_data({"fractal dimensions": _data})
        # print(avg_frac_dim, regress)

        # file_exists = pathlib.Path("fractal_dimensions.dat").is_file()
        # with open("fractal_dimensions.dat", "a") as f:
        #     if not file_exists:
        #         f.write(f"time    average_fractal_dimension    slope    R2    curve\n")
        #     f.write(
        #         f"{self.time}    {avg_frac_dim}    {regress[0]}    {regress[1]}    {regress[2]}\n"
        #     )


class Tbox(pyFLASH):
    def __init__(self, whoami="pyFLASH::Tbox"):
        self._whoami = whoami
        super().__init__(self._whoami)

    @timer
    def chooseTurb(self, iturb):

        self.turb = "driven"

        self.win_domain = self.win_right - self.win_left
        self.win_dims = (self.win_domain / self.minDelta).astype(np.int64)

        print("Window Left Edge:  ", self.win_left)
        print("Window Right Edge: ", self.win_right)
        print("Window Domain:     ", self.win_domain)
        print("Window Dimensions: ", self.win_dims)

    @timer
    def run(self, bname, bdir):

        self.bname = bname
        self.bdir = bdir
        self.FLASH_props()
        self.win_var = "c12 "
        self.win_var_scale = 1e0

        import time

        # ti = time.time()
        # self.cube = self.ds.covering_grid(
        #     self.lrefMax, left_edge=self.left, dims=self.maxDim
        # )

        self.save_scalars()

        self.getRadialReynoldStress(find_window=False)

        turbs = [0]
        for iturb in turbs:
            self.getWindow()
            self.chooseTurb(iturb)
            self.getCube()
            self.getIndices("dens", 1e0)
            self.set_velocity_data()
            self.multi_fractal_dimension("rpv1")
            self.fractal_dimension("rpv1", 0.99)
            self.structure_functions()
            self.getKESpectra()

            del [self.cube]
            del [self.indices]

        self.FLASH_close()


class Fbox(pyFLASH):

    def __init__(self, whoami="pyFLASH::Fbox"):
        self._whoami = whoami
        super().__init__(
            self._whoami,
        )

    @timer
    def getWindow(self):

        self.no_win = True

        self.win_left = np.copy(self.left)
        self.win_right = np.copy(self.right)

        self.win_left[0] = self.win_xmin
        self.win_right[0] = self.win_xmax
        self.win_domain = self.win_right - self.win_left
        self.win_dims = (self.win_domain / self.maxDelta).astype(np.int64)
        self.no_win = False
        return

    @timer
    def getIndices(self, win_var, win_var_scale):

        self.win_var = win_var
        self.win_var_scale = win_var_scale

        try:
            _win_var = self.win_var_scale * self.cube[self.win_var].d
        except:
            _win_var = self.win_var_scale * self.cube["flam"].d

        self.indices = {}

        self.indices["Domain"] = _win_var < np.inf

        # if not self._get_indices:
        #     del[_win_var]
        #     return

        eps_pure = 1e-5
        self.indices["Flame_Threshold_1e0"] = np.copy(_win_var) <= 1e0
        self.indices["Flame_Threshold_9.99e-1"] = np.copy(_win_var) <= 9.99e-1
        self.indices["Flame_Threshold_9.9e-1"] = np.copy(_win_var) <= 9.9e-1
        self.indices["Flame_Threshold_9e-1"] = np.copy(_win_var) <= 9e-1
        self.indices["Flame_Threshold_7e-1"] = np.copy(_win_var) <= 7e-1
        self.indices["Flame_Threshold_5e-1"] = np.copy(_win_var) <= 5e-1
        self.indices["Flame_Threshold_3e-1"] = np.copy(_win_var) <= 3e-1
        self.indices["Flame_Threshold_1e-1"] = np.copy(_win_var) <= 1e-1
        self.indices["Flame_Threshold_1e-2"] = np.copy(_win_var) <= 1e-2
        self.indices["Flame_Threshold_1e-3"] = np.copy(_win_var) <= 1e-3
        self.indices["Flame_Threshold_1e-4"] = np.copy(_win_var) <= 1e-4
        self.indices["Flame_Threshold_1e-5"] = np.copy(_win_var) <= 1e-5
        self.indices["PureFuel"] = np.copy(_win_var) < eps_pure
        self.indices["PureAsh"] = np.copy(_win_var) > 1 - eps_pure

        #        self.indices["Ash"] = _win_var >= 0.99

        #        self.indices["Flame"] = (_win_var < 0.99) & (_win_var > 0.01)

        del [_win_var]

    @timer
    def chooseTurb(self, iturb):

        self.turb = "Flame Window"

        self.win_domain = self.win_right - self.win_left
        self.win_dims = (self.win_domain / self.minDelta).astype(np.int64)

        print("\n\nWindow Left Edge:  ", self.win_left / 1e5)
        print("Window Right Edge: ", self.win_right / 1e5)
        print("Window Domain:     ", self.win_domain / 1e5)
        print("Window Dimensions: ", self.win_dims, "\n\n")

    @timer
    def run(self, bname, bdir):

        self.bname = bname
        self.bdir = bdir
        self._analysis_base = self.bdir / self._analysis_base
        self.FLASH_props()
        self.win_var = "rpv1"
        self.win_var_scale = 1e0

        self.save_scalars()
        # self.cube = self.ds.covering_grid(
        #     self.lrefMax, left_edge=self.left, dims=self.maxDim
        # )
        self.getRadialReynoldStress()

        turbs = [0]
        for iturb in turbs:
            self.getWindow()
            self.chooseTurb(iturb)
            self.getCube()
            self.set_velocity_data()
            self.getIndices("rpv1", 1e0)
            self.multi_fractal_dimension("rpv1")
            self.fractal_dimension("rpv1", 0.99)
            self.structure_functions()
            self.velocity_rms()

            bounds = {
                "temp": [1e9, 3e9],
                "dens": [6.4, 6.8],
                "pres": [0, 2.5],
                "igtm": [-4, 4],
                "vort": [0, 500],
                "divv": [-50, 50],
                "mach": [0.0, 0.2],
                "velxFuelWeighted": [-500e5, 500e5],
                "velxFlameWeighted": [-500e5, 500e5],
                "vortFuelWeighted": [0, 500],
                "vortFlameWeighted": [0, 500],
                "divvFuelWeighted": [-50, 50],
                "divvFlameWeighted": [-50, 50],
                "igtmFuelWeighted": [-4, 4],
                "igtmFlameWeighted": [-4, 4],
                "presFuelWeighted": [0, 2.5],
                "presFlameWeighted": [0, 2.5],
                "densFuelWeighted": [6.4, 6.8],
                "densFlameWeighted": [6.4, 6.8],
                "velx_dot_velx": [0, 2.5e15],
                "velx_dot_vely": [-8e14, 8e14],
                "velx_dot_velz": [-1e15, 1e15],
                "vely_dot_velx": [-8e14, 8e14],
                "vely_dot_vely": [0, 1.5e15],
                "vely_dot_velz": [-8e14, 8e14],
                "velz_dot_velx": [-1e15, 1e15],
                "velz_dot_vely": [-8e14, 8e14],
                "velz_dot_velz": [0, 1.5e15],
                "velx_dot_velxFlameWeighted": [0, 2.5e15],
                "velx_dot_velyFlameWeighted": [-8e14, 8e14],
                "velx_dot_velzFlameWeighted": [-1e15, 1e15],
                "vely_dot_velxFlameWeighted": [-8e14, 8e14],
                "vely_dot_velyFlameWeighted": [0, 1.5e15],
                "vely_dot_velzFlameWeighted": [-8e14, 8e14],
                "velz_dot_velxFlameWeighted": [-1e15, 1e15],
                "velz_dot_velyFlameWeighted": [-8e14, 8e14],
                "velz_dot_velzFlameWeighted": [0, 1.5e15],
                "velx_dot_velxFuelWeighted": [0, 2.5e15],
                "velx_dot_velyFuelWeighted": [-8e14, 8e14],
                "velx_dot_velzFuelWeighted": [-1e15, 1e15],
                "vely_dot_velxFuelWeighted": [-8e14, 8e14],
                "vely_dot_velyFuelWeighted": [0, 1.5e15],
                "vely_dot_velzFuelWeighted": [-8e14, 8e14],
                "velz_dot_velxFuelWeighted": [-1e15, 1e15],
                "velz_dot_velyFuelWeighted": [-8e14, 8e14],
                "velz_dot_velzFuelWeighted": [0, 1.5e15],
            }
            self.getPDF1d(bounds.keys(), bounds)
            # ti = time.time()
            # for keys,vals in bounds.items():
            #     self.getPDF1d([keys], {keys:vals})
            # # self.getPDF1d(["velx_dot_velx", "velx_dot_vely", "velx_dot_velz", "vely_dot_velx", "vely_dot_vely", "vely_dot_velz", "velz_dot_velx", "velz_dot_vely", "velz_dot_velz"])
            # # self.getPDF1d(["velx_dot_velxFlameWeighted", "velx_dot_velyFlameWeighted", "velx_dot_velzFlameWeighted", "vely_dot_velxFlameWeighted", "vely_dot_velyFlameWeighted", "vely_dot_velzFlameWeighted", "velz_dot_velxFlameWeighted", "velz_dot_velyFlameWeighted", "velz_dot_velzFlameWeighted"])
            # # self.getPDF1d(["velx_dot_velxFuelWeighted", "velx_dot_velyFuelWeighted", "velx_dot_velzFuelWeighted", "vely_dot_velxFuelWeighted", "vely_dot_velyFuelWeighted", "vely_dot_velzFuelWeighted", "velz_dot_velxFuelWeighted", "velz_dot_velyFuelWeighted", "velz_dot_velzFuelWeighted"])
            # #  self.getPDF1d(["pres", "densFuelWeighted", "densFlameWeighted"])
            # print("Get 1D PDF: {0:0.4f}".format(time.time()-ti))

            bounds = {
                "velx": [-500e5, 500e5],
                "vely": [-400e5, 400e5],
                "velz": [-400e5, 400e5],
            }
            self.getTurbVelStats(bounds.keys(), bounds)
            self.getKESpectra()

            del [self.cube]
            del [self.indices]

        self.FLASH_close()
