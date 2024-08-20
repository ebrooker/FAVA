

Flash Model is composed of

    1. HDF5 files (primary results)
        - hdf5_chk_xxxx
            - All information on fluid
            - Contains scalar data
            - Contains field data (dens, pres, temp, etc)
            - All information on particles if they exist
        - hdf5_plt_cnt_xxxx
            - Selected information on fluid
            - Contains scalar data
            - Contains field data (dens, pres, temp, etc)
        - hdf5_part_xxxx
            - Information on particles
            - Contains scalar data
            - Contains field data (dens, pres, temp, etc)

    2. Columnated ASCII text files (secondary results)
        - .dat file (always)
            - Contains volume integrated data of selected fields and derived fields
        - .log file (always)
            - Runtime log information
        - .alp files (optional; Average Lateral Profile)
            - Contains lateral averaged data of selected fields and derived fields

Other Simulation Models may be composed of different types of results files, but that is OK. We can let inheritance and registries manage that

Essentially, we have a data heirarchy of...

    Model
        Results
            Scalars
            Coordinates
            Fields
            Derived Fields






Mesh abstract base class. Inherited by Structured/Unstructured/???

Meshes must have a mapping for data field names of a specific mesh that are tied to a generalized name.

Structured Inherited by FlashAmr/FlashUniform



We load a mesh. There will be an algorithm to determine precedence for what loader is used.

We register meshes with their unique loaders for the LOAD function to choose from.



We register analysis functions that operate over generic data structures. However, they must know the names of the data that they select from datasets.

These analysis functions must support parallel operations on their own.

Can we use the register functions' names? Or must we use a calling function like the dataset LOAD circumstance? (RUN_ANALYSIS as a name?)


We also want to be able to plot various things.

These plot procedures will be dynamically registered as well.

They will be called with the PLOT method


Therefore we need the following decorators:

    register_mesh
    register_analysis
    register_plot

These decorators will have the 'overwrite' keyword that is default False.
If the function has been registered, to register a new version of the same name one must pass overwrite as True


We need to have a method of representing data, whether it is distributed as chunks of N-dimensional data or just
straight N-dimensional arrays. For instance, FlashAMR data is clearly chunks of {1,2,3}-d data patched together, 
but a uniform mesh could be one "chunk" of {1,2,3}-d data.