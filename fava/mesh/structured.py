from fava.model import Model
from fava.mesh.mesh import Mesh

@Model.register_mesh()
class Structured(Mesh):
    """
    Base implementation for structured meshes.
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    