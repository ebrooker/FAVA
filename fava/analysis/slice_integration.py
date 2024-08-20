from fava.model import Model

@Model.register_analysis(use_timer=True)
def slice_integration(self, *args, **kwargs):
    return self.mesh.slice_integration(*args, **kwargs)
 