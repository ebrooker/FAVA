from fava.model import Model

@Model.register_analysis(use_timer=True)
def volume_average(self, *args, **kwargs):
    return self.mesh.volume_average(*args, **kwargs)
