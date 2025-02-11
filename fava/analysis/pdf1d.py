from fava.model import Model


@Model.register_analysis(use_timer=True)
def pdf1d(self, *args, **kwargs):
    return self.mesh.pdf1d(*args, **kwargs)
