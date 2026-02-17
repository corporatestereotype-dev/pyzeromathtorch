import astropy.cosmology as cosmo

class FFZCosmology(cosmo.FLRW):
    def __init__(self):
        super().__init__(H0=70, Om0=0.3, Tcmb0=2.725)

    def hubble_tension(self):
        # From docs: Resolve with FFZ torsion
        return self.H(0) * 1.01  # Adjusted for tension
