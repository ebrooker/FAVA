====
FAVA
====

Fluids Advanced Variable Analysis: package for performing advanced analysis calculations on computational fluid dynamics data.

* Free software: MIT license
* Documentation: https://FAVA.readthedocs.io.

Usage
-----

One can create a `flash` type object that takes in the directory path to FLASH structured HDF5 data files. This will create an interface to iteratively loading and analyzing a set of data files, enabling easier development and configuration of analysis pipelines.

.. code-block:: Python
import fava
from pathlib import Path

model_dir = Path("/path/to/directory/with/FLASH/data")
model = fava.flash(model_dir)
```

The program will auto-detect the presence of checkpoint, plot, and particle files, organizing them in a class attribute master dictionary by indexing value and output file number (both sets are organized by `chk`, `plt`, and `part` files). Additionally, the code will identify existing `uniform` and `analysis` type files designed for use with FLASH type data in FAVA.


Features
--------

* TODO

Credits
-------

This package was initially created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template with modifications.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage