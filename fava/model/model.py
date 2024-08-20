import copy
import h5py

from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

from fava.util._exceptions import NotCallableError
from fava.util import timer

class Model:
    """Class encapsulating a data model. Requires a directory path to the model's data and optionally a model name.

    The Model class can handle identifying the model format
    """

    __meshes: Dict[str, Any] = {}
    _directory: Path
    _name: str
    _frontend: str = "Generic"

    def __init__(self, directory: str | Path, name: str = None):

        self.directory = Path(directory)
        self.name = name

    @property
    def directory(self) -> Path:
        """Returns a string of the model directory Path object

        Returns:
            str: model directory string
        """
        return self._directory

    @directory.setter
    def directory(self, directory: str | Path):
        """Sets the directory path attribute and ensures that it is a Path object

        Args:
            directory (str | Path): Directory Path for the model
        """
        self._directory = Path(directory)
        if not self._directory.is_dir():
            raise FileNotFoundError(f"Cannot find model directory: {self._directory}")

        self.files = sorted(fn for fn in self._directory.glob("*") if fn.is_file())
        if len(self.files) == 0:
            raise FileNotFoundError(f"The model directory is empty: {self._directory}")

    @property
    def name(self) -> str:
        """Returns model name string

        Returns:
            str: The model name
        """
        return self._name

    @name.setter
    def name(self, name: str | None):
        """Sets the name of the model, if the parameter is None, we default to the directory name

        Args:
            name (str | None): string name of the model (can be None)
        """
        self._name = self._directory.name if name is None else name

    def _filter_files(self, pattern: str):
        return [file for file in self.files if file.match(pattern) ]


    def nfiles(self, *args, **kwargs) -> int:
        return len(self.files)

    # --------------------------
    # Register and load meshes
    @classmethod
    def register_mesh(cls):
        def decorator(mesh_cls):
            cls.__meshes[mesh_cls.__name__] = mesh_cls
            return mesh_cls

        return decorator

    @classmethod
    def mesh_names(cls) -> list:
        """ Returns the sorted list of mesh names registered """
        return sorted(cls.__meshes.keys())

    def _load_mesh(self, filename: str | Path, fields: Optional[List[str]] = None):
        """
        _mesh = self.__meshes.get(self.frontend)

        if _mesh is None:
            raise Exception(f"The frontend {self.frontend} has not been registered")

        try:
            self.mesh = _mesh(filename)
            self.mesh.load(fields)
        except Exception as exc:
            self.mesh = None
            print(exc)
        """
        pass

    def load(self, filenumber: int=0):
        """
        if len(self.files) <= filenumber:
            raise IndexError(f"Filenumber {filenumber} is out of bounds for filelist of length {len(self.files)}")

        self._load_mesh(str(self.files[filenumber]))
        """
        pass


    # --------------------------
    # Register analysis methods
    @classmethod
    def register_analysis(cls, overwrite=False, use_timer=None):
        def decorator(analysis_func):
            if not callable(analysis_func):
                raise NotCallableError(analysis_func)
            name = analysis_func.__name__
            if not hasattr(cls, name) or overwrite:
                if use_timer:
                    setattr(cls, name, timer(analysis_func))
                else:
                    setattr(cls, name, analysis_func)
            return analysis_func

        return decorator

    # --------------------------


    # --------------------------
    # HDF5 API methods

    def save_to_hdf5(self, data: dict, filename: Path | str):
        _filename = Path(filename)

        # Save the data to the analysis file, append mode if it already exists
        mode = "a" if _filename.is_file() else "w"
        with h5py.File(str(_filename), mode) as f:
            # Start recursive writing of the data dictionary
            self.write_to_hdf5(f, data)

    def write_to_hdf5(self, handle, data):
        # Iterate through the items in the data dictionary
        for key, values in data.items():

            # If the values for this key is another dictionary, we need to go one group level down
            if isinstance(values, dict):
                # Create a group handle and pass it and the sub-dict to write_to_hdf5()
  
                try:
                    group = handle.create_group(key)
                except:
                    group = handle[key]
                self.write_to_hdf5(group, values)

            # No more sub-groups we can write the dataset now.
            else:

                overwrite = key in handle.keys()

                try:
                    _ = values.copy()
                    if overwrite:
                        del handle[key]

                    handle.create_dataset(key, data=values.copy())

                except Exception as exc1:
                    print(exc1)
                    try:
                        _ = copy.copy(values)
                        if overwrite:
                            del handle[key]

                        handle.create_dataset(key, data=copy.copy(values))

                    except Exception as exc2:
                        print(exc2)
                        print(f"[ERROR] in making {key} for {handle}")


    def hdf5_key_exists(self, key: str, filename: str | Path) -> bool:
        _filename = Path(filename)
        exists: bool = False
        if _filename.is_file():
            with h5py.File(str(_filename), "r") as f:
                exists = key in list(f.keys())
        return exists

    # --------------------------

