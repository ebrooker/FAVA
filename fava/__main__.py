from pathlib import Path
from fava import Model, FLASH

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib.colors import Normalize

if __name__ == "__main__":

    # p = Path("/Users/ezrabrooker/SimulationData/mnt/nersc/rtflame/3d/rtfD3x512d66t159g119c42p/")
    p = Path("/Users/ezrabrooker/SimulationData/mnt/test_data")

    m = Model(p)

    print(Model.mesh_names())
    print(FLASH.mesh_names())

    import numpy as np
    from math import floor

    f = FLASH(p)

    filenums = sorted(f.prt_files.get("by number", []))
    pbeg = filenums[0]
    pend = filenums[-1]
    fnums = range(pbeg, pend+1)
    imid = floor(len(fnums)/2)
    tvar='igtm'
    svar='divv'
    fields = [tvar, 'flam', 'posx', 'posy', 'posz', 'tag']

    poi = np.array([2.02875e7, 1.4625e6 ,1.4875e6]) # 128 pt0
    # poi = np.array([335.6e5,  5.0e5, -7.8e5])  # 512 pt 1

    f.load(file_index=imid, fields=fields, file_type="prt")
    
    pcrds = f.particles.get_coords()
    print("Number of particles: ", f.particles.data['tag'].size)
    fuel = f.particles.data["flam"] <= 0.01

    pcrds -= poi
    pcrds = np.sum(pcrds**2, axis=1)
    min_dist = pcrds[fuel].min()

    poi_idx = np.where(pcrds == min_dist)[0]

    crds = f.particles.get_coords()
    rcrd2 = (crds[:,0] - poi[0])**2

    if f.particles.ndim == 2:
        roi = 2e5
    else:
        roi = 2e5
    roi2 = roi*roi

    if f.particles.ndim > 1:
        poi[1] = f.particles.data["posy"][poi_idx]
        rcrd2 += (crds[:,1] - poi[1])**2

    if f.particles.ndim > 2:
        poi[2] = f.particles.data["posz"][poi_idx]
        rcrd2 += (crds[:,2] - poi[2])**2

    rmask = rcrd2 <= roi2
    rmask = np.where(rmask)[0]
    samps_c = crds[rmask,:]

    samps = f.particles.data["tag"][rmask]
    fuel = f.particles.data["flam"][rmask]
    poi_idx = f.particles.data['tag'][poi_idx].astype(int)

    rho = f.cross_correlation(svar, tvar, samps, poi_idx, lagrangian_tracking=True, tag_field="tag", middle_time_point=imid, file_type="prt")


    field = "temp"
    f.load(file_number=1252, file_type="plt_prt")
    blklist = f.mesh.get_list_of_blocks(blkType="LEAF")
    dmin = np.inf
    dmax = -np.inf

    for lb, blk in enumerate(blklist):
        Z = f.mesh.data[field][blk,...]

        dmax = max(dmax, Z.max())
        dmin = min(dmin, Z.min())

    cmx = plt.get_cmap("RdBu")
    cNorm = Normalize(vmin=rho.min(), vmax=rho.max())
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmx)
    fig = plt.figure(figsize=(5,1), dpi=300)
    ax = fig.add_subplot(projection='3d')

    fmask = fuel < 1.0
    fmask = f.particles.data[tvar][rmask] < 1e1
    x,y,z = samps_c[:,0]/1e5, samps_c[:,1]/1e5, samps_c[:,2]/1e5

    x = f.particles.data[svar][rmask][fmask]
    y = np.log10(f.particles.data[tvar][rmask])[fmask]
    z = np.log10(f.particles.data["flam"][rmask])[fmask]
    rho1 = rho[fmask]

    fmask = z < np.inf
    ax.scatter(x[fmask], y[fmask], z[fmask], c=scalarMap.to_rgba(rho1[fmask]), s=1)
    ax.set_xlabel(svar)
    ax.set_ylabel(tvar)
    ax.set_zlabel("flam")
    pltfn =  f.directory / f"crosscorr_3d_part_{tvar}_{svar}_{poi[0]}_{poi[1]}_plt{pbeg}_plt{pend}.png"
    
    scalarMap.set_array(rho1[fmask])
    # fig.colorbar(scalarMap, ax=ax)
    fig.set_size_inches(20,2)
    plt.tight_layout()
    plt.show()
    plt.close('all')
    # plt.savefig(str(pltfn), dpi=1024)

    
    t, r = f.lagrangian_autocorrelation(nsamples=100, fields=["velx", "vely"])
    # t, r = f.eulerian_autocorrelation(nsamples=1000, fields=["velx", "vely"])