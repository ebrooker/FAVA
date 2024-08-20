from fava.model import Model

@Model.register_analysis(use_timer=True)
def volume_integration(self, *args, **kwargs):
    return self.mesh.volume_integration(*args, **kwargs)
