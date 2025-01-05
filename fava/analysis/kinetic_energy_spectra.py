from fava.model import Model


@Model.register_analysis(use_timer=True)
def kinetic_energy_spectra(self, *args, **kwargs):
    return self.mesh.kinetic_energy_spectra(*args, **kwargs)
