from fava.model import Model


@Model.register_analysis(use_timer=True)
def structure_functions(self, *args, **kwargs):
    return self.mesh.structure_functions(*args, **kwargs)
