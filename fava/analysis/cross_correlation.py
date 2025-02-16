
import numpy as np

from math import floor
from typing import List

from fava.model import Model

@Model.register_analysis(use_timer=True)
def cross_correlation(self, spatial_field: str, temporal_field: str, sample_points: np.ndarray, poi_idx: int, *args, **kwargs):
    """Spatio-temporal cross correlation method detailed in Naka et al. 2015 Space-time pressure-velocity correlations boundary layer turbulence

    Computes the cross correlation values of a set of spatial points against a single point of interest across time.

    The general idea is that the cross correlation is computed for each sample point at time t against point of interest
    at time t+dt. We can correlate any two field measures, including autocorrelating fields.

    The correlation process will operate over a series of files operating under the assumption that the point of interest
    was identified at the floor(number_of_files / 2) point in the list of files. This creates a time-centered correlation
    that allows us to see how the surrounding media influence the temporal field measure before and after the point of
    interest was identified.

    Arguments:
        spatial_field (str): String value giving the field measurement name for the spatial set of sample points

        temporal_field (str): String value giving the field measurement name for the point of interest across time

        sample_points (np.ndarray): Number array of the sample points used for cross correlation. In the case of the
                                    Lagrangian tracking mode, we want sample_points to be an array particle tag IDs.
                                    This is modeled after the FLASH particle data structure.

        poi_idx (int): In Lagrangian tracking mode, this is an integer tag ID of the particle that tracks the point of
                        interest.

        *args: Usual list of input values

        **kwargs: Usual dictionary of keyword arguments. For now we are passing "lagrangian_tracking" as a kwarg.

    Results:
        rho (np.ndarray[sample_points.size]): Numpy array (floats) sized by the number of sample points being used for
                                              the cross correlation process. This represents the normalized cross
                                              correlation coefficients of each sample point against the point of interest.
    """
    
    tvar: str = temporal_field
    svar: str = spatial_field
    fields: List[str] = [svar, tvar]

    nfiles: int = len(self.prt_files["by index"])
    npts: int = sample_points.size

    ibeg: int = 0
    ibeg = kwargs.get("ibeg", ibeg)

    iend: int = nfiles
    iend = kwargs.get("iend", iend)

    imid: int = floor((iend-ibeg)/2)

    lagrangian_tracking = kwargs.get("lagrangian_tracking")
    if lagrangian_tracking is not None:
        
        tagvar = kwargs.get("tag_field")
        if tagvar is None:
            raise Exception("Lagrangian Particle tracking has been selected but no name has been given for accessing Particle ID tags in data strucuture")

        # Initialize data arrays
        samp_data = np.zeros((nfiles, npts), dtype=float)
        temp_data = np.zeros((nfiles, 1), dtype=float)

        #Load the midpoint file and get the temporal data measure at this time to start correlating
        self.load(file_index=imid, fields=[*fields, tagvar], *args, **kwargs)

        # Get the temporal measure tag(s) and midpoint temporal data
        temp_tags = np.where(self.particles.data[tagvar] == poi_idx)[0]
        temp_data[imid] = self.particles.data[tvar][temp_tags]

        # Get the spatial measure tag(s) and midpoint spatial data
        samp_tags = np.squeeze(np.array([np.where(self.particles.data[tagvar] == smp)[0] for smp in sample_points], dtype=int))
        samp_data[imid,:] = self.particles.data[svar][samp_tags]

        # Obtain the spatial and temporal data for all remaining time slices
        for i in range(nfiles):
            print(f"{i=}")
            if i == imid:
                continue
            self.load(file_index=i, fields=fields, *args, **kwargs)
            temp_data[i] = self.particles.data[tvar][temp_tags]
            samp_data[i,:] = self.particles.data[svar][samp_tags]

    else:
        return None

    # Compute mean for spatial and temporal measures
    smean = samp_data[:-1,...].mean(axis=0)
    tmean = temp_data[1:].mean()

    # Compute stddev for spatial and temporal measures
    sstd = samp_data[:-1,...].std(axis=0)
    tstd = temp_data[1:].std()

    # Compute cross correlations normalized over number of time slices
    Rts = np.sum(temp_data[1:] * samp_data[:-1,:], axis=0) / float(nfiles-1)

    # Subtract out the cross correlation mean (I think this forms the cross covariance matrix)
    Kts = Rts - smean*tmean

    # Compute the statistically normalized cross correlation matrix
    rho = Kts / (sstd*tstd)

    return rho
