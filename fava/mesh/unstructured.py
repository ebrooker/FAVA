from fava.temporary import Temporary
from fava.mesh import Mesh

@Temporary.register_mesh()
class Unstructured(Mesh):

    def __init__(self):
        super().__init__()
