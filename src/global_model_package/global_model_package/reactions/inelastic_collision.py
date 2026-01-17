from typing import override
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from global_model_package.specie import Specie, Species
from .reaction import Reaction
from global_model_package.chamber_caracteristics import Chamber


class InelasticCollision(Reaction):
    """
        Represents the collisions between ions accelerated towards the walls and the neutral gas.
        Works for 3 temperatures : that of electrons, monoatomic and diatomic particles

    """

    def __init__(self, 
                 species: Species, 
                 chamber: Chamber
                 ):
        """
        Represents the collisions between ions accelerated towards the walls and the neutral gas.
        Takes all ions and all neutrals into account.
        Works for 3 temperatures : that of electrons, monoatomic and diatomic particles
        
        Inputs : 
            species : instance of class Species, lists all species present 
            chamber : instance of class Chamber, contains the chamber characteristics
        """
        self.charged_sp = [sp for sp in species.species if sp.charge != 0]
        self.non_charged_sp = [sp for sp in species.species if sp.charge == 0]
        # species.names[0] nom des électrons
        super().__init__(species, species.names[1:], species.names[1:], chamber)
        
    @override
    def density_change_rate(self, state: NDArray[float]): # type: ignore
        """Returns an np.array with the change rate for each species due to this reaction
        state has format : [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato]"""
        rate = np.zeros(self.species.nb)
        
        return rate

    
    @override
    def energy_change_rate(self, state):
        rate = np.zeros(3)
        E_ion=0

        for sp in self.charged_sp:
            if sp.nb_atoms > 0:
                E_ion += sp.mass * state[sp.index] * self.chamber.u_B(state[self.species.nb], sp.mass)**2
        
        for sp in self.non_charged_sp:
            rate[sp.nb_atoms] += self.chamber.gamma_neutral(state[sp.index], state[self.species.nb + sp.nb_atoms], sp.mass) * E_ion * self.chamber.SIGMA_I

        self.var_tracker.add_value_to_variable_list('energy_change_inelastic_collision', rate)

        return rate

























