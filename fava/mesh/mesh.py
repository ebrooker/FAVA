
from abc import ABC, abstractmethod

class Mesh(ABC):
    """Base Mesh class that implements the generic framework for grid meshes"""

    _fields = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

    @property
    def mesh_type(self):
        return self.__class__.__name__

    @property
    def fields(self):
        notmapped = "NotMapped"
        s = "".join(f"\n|| {k:24} -> {v:24}" for k,v in self._fields.items() if k != notmapped)
        tmp = ""
        if notmapped in self._fields:
            tmp = ", ".join(v for v in self._fields[notmapped])
            tmp = f"\n||\n|| Not Mapped: [ {tmp} ]"
        return s + tmp