from typing import override, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.constants import m_e, e, pi, k as k_B, epsilon_0 as eps_0, mu_0   # k is k_B -> Boltzmann constant

from global_model_package.specie import Specie, Species
from .reaction import Reaction
from global_model_package.chamber_caracteristics import Chamber

class Dissociation(Reaction):
    """
    Elastic collision between a particle and an electron. Energies should be in eV
    Works with 3 temperatures : Te, Tmono, Tdiat
    In reactives, electron must be in first position and colliding_specie next.
    """
    instance_counter = {}
    def __init__(self, species: Species, diatomic_reactive: str, monoatomic_product: str, rate_constant: Callable, energy_threshold: float, monoatomic_energy_excess: float, chamber: Chamber):
        """ Dissociation class
        Parameters:
        -------------
        species : Species
            Instance of class Species, lists all species present.
        diatomic_reactive : str
            Name of molecular reactive involved in the dissociation reaction.
        monoatomic_product : str
            Name of molecular product formed after the dissociation reaction. Should NOT contain the electron.
        rate_constant : Callable
            Function taking as argument the state [n_e, n_N2, ..., n_N+, T_e, T_monoato, ..., T_diato].
        energy_threshold : float
            Energy threshold of the electron so that the reaction occurs.
        monoatomic_energy_excess : float
            Energy excess of one monoatomic product after dissociation.
        chamber : Chamber
            Instance of the Chamber class containing chamber characteristics.
        """
        super().__init__(species, [species.names[0], diatomic_reactive], [species.names[0], monoatomic_product], chamber)
        self.rate_constant = rate_constant
        self.energy_threshold = energy_threshold
        self.monoatomic_energy_excess = monoatomic_energy_excess
        
        name = f"{diatomic_reactive}"
        if name in self.instance_counter:
            self.name = f"diss_{name}_{self.instance_counter[name]}"
            self.instance_counter[name] += 1
        else:
            self.name = f"diss_{name}_0"
            self.instance_counter[name] = 1

    @override
    def density_change_rate(self, state):
        rate = np.zeros(self.species.nb)
        K = self.rate_constant(state)
        sp_diato, sp_monoato = self.reactives[1], self.products[1]

        reaction_speed = K * np.prod(state[self.reactives_indices])
        rate[sp_diato.index] -=  reaction_speed
        rate[sp_monoato.index] += 2 * reaction_speed

        self.var_tracker.add_value_to_variable("D_"+self.name, reaction_speed)

        return rate
    
    @override
    def energy_change_rate(self, state):  # TODO : Rajouter les termes de perte d'énergie dus a la baisse de densité de l'espèce diatomique
        rate = np.zeros(3)

        K = self.rate_constant(state)
        reac_speed = K * np.prod(state[self.reactives_indices])
        
        rate[0] = - reac_speed * e * self.energy_threshold
        
        rate[1] = 2 * e * self.monoatomic_energy_excess * reac_speed 

        self.var_tracker.add_value_to_variable("E_e-_"+self.name, reac_speed * self.energy_threshold)
        self.var_tracker.add_value_to_variable("E_mono_"+self.name, 2 * self.monoatomic_energy_excess * reac_speed )
        
        return rate