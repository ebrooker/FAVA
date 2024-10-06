Analyses to add to this package:

    - mass-weighted 1D PDFs
        - we don't need a uniform mesh for this, just dens * cell_volume for weights

    - mass-weighted 2D PDFs
        - same deal as 1D PDFs

    - structure functions
        - do not need a uniform grid as we are just doing sampled velocity differences
        - a uniform grid only changes things if smooth the parts that get refined, but that modifies the data

    - Helmholtz decomposition
        - this will require implementing a multigrid method


Other changes

    - when initializing an analysis model we should consider the following adaptations to the current consider
        - Specify the model type, e.g., FLASH, Enzo, etc
        - Provide the files to be analyzed directly, especially for above user specification
        - Support including different types of data files
            - E.g., FLASH has Eulerian grid data, particle data, statistics files, integrated quantities, average lateral profiles, etc


    - Support for adding derived fields to the internal data attribute of a model when loaded in similar to yt

    - Genericize more of the algorithms so they don't have to be implemented for each mesh/model type
        - Complementary to this, create genericized views of data structures when possible
            - E.g., FLASH stores data as [blk, i, j, k], flatten the array into [i, j, k] dimensions with the view
        - np.view(some_array).reshape(some_shape) might be what I want to use
        - ensure the that coordinate data is accessible in the same way as the data views
