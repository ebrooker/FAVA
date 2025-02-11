====
FAVA
====

Fluids Advanced Variable Analysis: package for performing advanced analysis calculations on computational fluid dynamics data.

* Free software: MIT license
* Documentation: https://FAVA.readthedocs.io.

Usage
-----

One can create a <flash> type object that takes in the directory path to FLASH structured HDF5 data files. This will create an interface to iteratively loading and analyzing a set of data files, enabling easier development and configuration of analysis pipelines.

.. code-block:: Python


    import fava
    from pathlib import Path

    model_dir = Path("/path/to/directory/with/FLASH/data")
    model = fava.flash(model_dir)


The program will auto-detect the presence of checkpoint, plot, and particle files, organizing them in a class attribute master dictionary by indexing value and output file number (both sets are organized by `chk`, `plt`, and `part` files). Additionally, the code will identify existing `uniform` and `analysis` type files designed for use with FLASH type data in FAVA. These can be iterated over and compute various quantities such as the Reynolds stress profile over a given axis as follows:

.. code-block:: Python


    for i in sorted(pipe.model.plt_files["by index"].keys())
        model.load(file_index=i, file_type="plt")
        model.reynolds_stress(axis=0)  # axis=0 --> x in Cartesian

It should be noted that the methods implemented for a <flash> type object are defined in the fava/analysis and serve to be wrapper calls to the specific implementations for a given mesh-type. These wrapper methods are registered to a generic <model> class that the <flash> class inherits from.

One can also create a <mesh>-type object for different types of FLASH data, such as AMR, Uniform, and Particle (partial implementation) data. (These mesh objects are what get called internally by the above Reynolds stress example with the <flash> model object).

.. code-block:: Python


    from fava.mesh.FLASH import FLASH as FlashAMR
    from fava.mesh.FLASH import FlashUniform
    
    amr_file = "path/to/AMR-based/FLASH/datafile/rtflame_hdf5_plt_cnt_0000"
    uni_file = "path/to/uniform-like/FLASH/datafile/rtflame_hdf5_uniform_0000"

    amr = FlashAMR(amr_file)
    amr.load()

    # Subdomain coordinates
    sd = [[0.0, 32.0e5], [0.0, 32.0e5], [0.0, 32.0e5]]

    # Fields to save to new uniformly refined data file
    fields = ["dens", "pressure", "temperature"]

    # Generates a uniformly data file from existing AMR mesh, subdomain 
    # coordinates can be provided to chop down the domain, and refine refine_level
    # of -1 indicates to refine to finest possible level
    amr.from_amr(subdomain_coords=sd, fields=fields, filename=uni_file, refine_level=-1)

    uni = FlashUniform(uni_file)
    uni.load()
    uni.kinetic_energy_spectra()


Features
--------

* TODO

Credits
-------

This package was initially created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template with modifications.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage