from fava.model import Model


@Model.register_analysis(use_timer=True)
def pdf2d(self, *args, **kwargs):
    return self.mesh.pdf2d(*args, **kwargs)
