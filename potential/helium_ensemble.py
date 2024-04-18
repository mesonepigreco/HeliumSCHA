import cellconstructor as CC, cellconstructor.Phonons
import numpy as np
import sscha, sscha.Ensemble

try:
    import julia
       try:
           from julia.api import Julia
           jl = Julia(compiled_modules=False)
           import julia.Main
           julia.Main.include(os.path.join(os.path.dirname(__file__),
               "compute_potential.jl"))
           __JULIA_EXT__ = True
       except:
           # Install the required modules
           julia.install()
           try:
               julia.Main.include(os.path.join(os.path.dirname(__file__),
                   "compute_potential.jl"))
               __JULIA_EXT__ = True
           except Exception as e:
               warnings.warn("Julia extension not available.\nError: {}".format(e))
   except Exception as e:
       warnings.warn("Julia extension not available.\nError: {}".format(e))


# Prepare the new ensemble for the fast calculation
# of helium
class HeliumEnsemble(sscha.Ensemble.Ensemble):
    def compute_ensemble(self, *args, **kwargs):
        # Perform the calculation of the ensemble using the julia code
        self.has_stress = False

        energies, forces = julia.Main.compute_ensemble_py(self.xats)

        # Copy the results to the ensemble
        self.force_computed[:] = True
        self.forces[:, :, :] = forces
        self.energies[:] = energies
        
        self.init()

    def get_energy_forces(self, *argd, **kwargs):
        self.compute_ensemble()



