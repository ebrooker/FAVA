from fava.model import Model

@Model.register_analysis(use_timer=True)
def slice_average(self, *args, **kwargs):
    return self.mesh.slice_average(*args, **kwargs)
