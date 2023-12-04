
import numpy as np

from math import floor
from pathlib import Path
from typing import List

import fava

from fava.util import timer

@timer
def cross_correlation(spatial_field: str, temporal_field: str, filenames: List[str|Path], sample_points: np.ndarray, poi_idx: int, *args, **kwargs):
    """Spatio-temporal cross correlation method detailed in Naka et al. 2015

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

        filenames (List[str|Path]): List of filenames to draw data from, must be in ascending time order.

        sample_points (np.ndarray): Number array of the sample points used for cross correlation. In the case of the
                                    Lagrangian tracking mode, we want sample_points to be a mask array over particle
                                    IDs. This is modeled after the FLASH particle data structure.

        poi_idx (int): In Lagrangian tracking mode, this is an integer ID of the particle that tracks the point of
                        interest.

        *args: Usual list of input values

        **kwargs: Usual dictionary of keyword arguments. For now we are passing "lagrangian_tracking" as a kwarg.

    Results:
        rho (np.ndarray[sample_points.size]): Numpy array (floats) sized by the number of sample points being used for
                                              the cross correlation process. This represents the normalized cross
                                              correlation coefficients of each sample point against the point of interest.
    """

    tvar = temporal_field
    svar = spatial_field
    fields = [svar, tvar]

    nfiles = len(filenames)

    # Get midpoint of files for time-centering. We want to know how a point evolves
    imid = floor(nfiles/2)


    # If we are tracking lagrangian evolved points, make that check now. E.g. are we using something like FLASH particles or not
    lagrangian_tracking = kwargs.get("lagrangian_tracking")
    if lagrangian_tracking:


        # Raw cross correlation array
        Rts = np.zeros(sample_points.size)

        # Array of sample data points measurements
        samp_data = np.zeros((nfiles, sample_points.size), dtype=float)

        # Array of temporal data measurements
        temp_data = np.zeros(nfiles, dtype=float)

        # Load the midpoint file and get the temporal data measure at this time to start correlating
        mesh = fava.load_mesh(filename=filenames[imid], fields=[tvar])
        temp_data[imid] = mesh.data[tvar][poi_idx]

        # Iterate in reverse over the first half of files; this works well enough if using lagrangian-tracked
        # data points, e.g. particles data structure in FLASH
        for i,fn in enumerate(reversed(filenames[:imid])):
            k = imid - i - 1
            mesh = fava.load_mesh(filename=fn, fields=fields)
            temp_data[k] = mesh.data[tvar][poi_idx]
            samp_data[k,:] = mesh.data[svar][sample_points]
            Rts[:] += temp_data[k+1] * samp_data[k,:]

        # Repeat process but forward through second half of data files
        for i,fn in enumerate(filenames[imid+1:]):
            k = imid + i + 1
            mesh = fava.load_mesh(filename=fn, fields=fields)
            temp_data[k] = mesh.data[tvar][poi_idx]
            samp_data[k,:] = mesh.data[svar][sample_points]
            Rts[:] += temp_data[k] * samp_data[k-1,:]

    # For eulerian tracked data
    else:
        ...

    # Need mean for spatial and temporal measures
    tmean = temp_data[1:].mean()
    smean = samp_data[:-1,...].mean(axis=0)

    # Same with standard deviation
    tstd = temp_data[1:].std()
    sstd = samp_data[:-1,...].std(axis=0)

    # Normalize raw cross correlations over number of time slices
    Rts /= float(nfiles-1)

    # Subtract out the cross correlation mean (I think this is the cross covariance matrix)
    Kts = Rts - tmean*smean

    # Get the normalized cross correlation matrix.
    rho = Kts / (tstd*sstd)

    return rho
