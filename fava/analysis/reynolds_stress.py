from fava.model import Model

@Model.register_analysis(use_timer=True)
def reynolds_stress(self, *args, **kwargs):
    return self.mesh.reynolds_stress(*args, **kwargs)
