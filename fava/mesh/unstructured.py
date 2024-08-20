from fava.model import Model
from fava.mesh.mesh import Mesh

@Model.register_mesh()
class Unstructured(Mesh):
    """
    Base implementation for unstructured meshes.
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
