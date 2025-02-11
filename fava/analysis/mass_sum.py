from fava.model import Model


@Model.register_analysis(use_timer=True)
def mass_sum(self, *args, **kwargs):
    return self.mesh.mass_sum(*args, **kwargs)
