from fava.temporary import Temporary
from fava.mesh import Mesh

@Temporary.register_mesh()
class Structured(Mesh):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    