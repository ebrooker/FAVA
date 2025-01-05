from fava.model import Model


@Model.register_analysis(use_timer=True)
def fractal_dimension(self, *args, **kwargs):
    return self.mesh.fractal_dimension(*args, **kwargs)
