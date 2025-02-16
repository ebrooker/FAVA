
from abc import ABC

class Mesh(ABC):
    """Base Mesh class that implements the generic framework for grid meshes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments

    @classmethod
    def is_this_your_mesh(cls, *args, **kwargs) -> bool:
        return False

    @property
    def mesh_type(self) -> str:
        return self.__class__.__name__
