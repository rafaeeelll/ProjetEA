import numpy as np
from scipy.constants import e, k as k_B, m_e
from datetime import datetime

from global_model_package.reactions import (Excitation, Ionisation, Dissociation, 
                VibrationalExcitation, RotationalExcitation,
                ThermicDiffusion, InelasticCollision, ElasticCollisionWithElectron, 
                FluxToWallsAndThroughGrids, GasInjection,
                ElectronHeatingConstantRFPower
            )

from global_model_package.specie import Species, Specie
from global_model_package.constant_rate_calculation import get_K_func, ReactionRateConstant
try:
    from .msis_densities import get_msis_neutral_atmosphere
except ImportError:
    from msis_densities import get_msis_neutral_atmosphere

ReactionRateConstant.CROSS_SECTIONS_PATH = "../../../cross_sections"

DEFAULT_MSIS_DATE = datetime(2020, 1, 1, 12, 0, 0)
DEFAULT_MSIS_LAT = 0.0
DEFAULT_MSIS_LON = 0.0
DEFAULT_MSIS_F107 = None
DEFAULT_MSIS_F107A = None
DEFAULT_MSIS_AP = None
OUTLET_AREA_M2 = 0.01
EARTH_RADIUS_M = 6371e3
EARTH_MU = 3.986004418e14


def get_species_and_reactions(
    chamber,
    altitude,
    argon_injection_rate,
    lat=DEFAULT_MSIS_LAT,
    lon=DEFAULT_MSIS_LON,
    date=DEFAULT_MSIS_DATE,
    f107=DEFAULT_MSIS_F107,
    f107a=DEFAULT_MSIS_F107A,
    ap=DEFAULT_MSIS_AP,
    ion_seed=1e8,
    electron_seed=1e12,
    A_intake=None,
    eta_collection=0.4,
    orbital_speed=None,
    atm=None,
):

    species = Species([Specie("e", m_e, -e, 0, 3/2), Specie("Ar", 6.63e-26, 0, 1, 3/2), Specie("Ar+", 6.63e-26, e, 1, 3/2), Specie("N2", 4.65e-26, 0, 2, 5/2), Specie("N", 2.33e-26, 0, 1, 3/2), Specie("N2+", 4.65e-26, e, 2, 5/2), Specie("N+", 2.33e-26, e, 1, 3/2), Specie("O2+", 5.31e-26, e, 2, 5/2), Specie("O2", 5.31e-26, 0, 2, 5/2), Specie("O", 2.67e-26, 0, 1, 3/2), Specie("O+", 2.67e-26, e, 1, 3/2)])

    if atm is None:
        atm = get_msis_neutral_atmosphere(
            altitude_km=float(altitude),
            lat=float(lat),
            lon=float(lon),
            date=date,
            f107=f107,
            f107a=f107a,
            ap=ap,
        )

    if A_intake is None:
        A_intake = 1

    tau_c = A_intake / OUTLET_AREA_M2

    if orbital_speed is None:
        r_orbit = EARTH_RADIUS_M + altitude * 1e3
        orbital_speed = np.sqrt(EARTH_MU / r_orbit)

    initial_state_dict = {
        "e": electron_seed,
        "Ar" : 0,
        "Ar+" : 0,
        "N2": atm["N2"],
        "N": atm["N"],
        "N2+": ion_seed,
        "N+": ion_seed,
        "O2+": ion_seed,
        "O2": atm["O2"],
        "O": atm["O"],
        "O+": ion_seed,
        "T_e": 1.0,
        "T_mono": atm["T_eV"],
        "T_diato": atm["T_eV"],
    }

    # Compression des neutres
    initial_densities = []
    for sp in species.species:
        base_density = initial_state_dict[sp.name]
        if sp.charge == 0 and sp.name != "e":
            initial_densities.append(tau_c * base_density)
        else:
            initial_densities.append(base_density)
    initial_state = initial_densities + [
        initial_state_dict["T_e"],
        initial_state_dict["T_mono"],
        initial_state_dict["T_diato"],
    ]
    
    # Particle injection rate in 1/s:
    # Ndot_i = eta_collection * n_i * u_orbit * A_intake
    intake_capture = eta_collection * orbital_speed * A_intake
    injection_rates = np.zeros(species.nb)
    injection_rates[species.get_specie_by_name("N2").index] = intake_capture * atm["N2"]
    injection_rates[species.get_specie_by_name("N").index] = intake_capture * atm["N"]
    injection_rates[species.get_specie_by_name("O2").index] = intake_capture * atm["O2"]
    injection_rates[species.get_specie_by_name("O").index] = intake_capture * atm["O"]
    injection_rates[species.get_specie_by_name("Ar").index] = argon_injection_rate


#  ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
# N2
    exc1_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc1_N2"), 6.17, chamber)
    exc2_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc2_N2"), 7.35, chamber)
    exc3_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc3_N2"), 7.36, chamber)
    exc4_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc4_N2"), 8.16, chamber)
    exc5_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc5_N2"), 8.40, chamber)
    exc6_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc6_N2"), 8.55, chamber)
    exc7_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc7_N2"), 8.89, chamber)
    exc8_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc8_N2"), 12.50, chamber)
    exc9_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc9_N2"), 12.90, chamber)
    # -- N'existe pas a priori exc10_N2 = Excitation(species_list, "N2", get_K_func(species_list, "N2", "exc10_N2"), 12.10, chamber)
    exc11_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc11_N2"), 12.90, chamber)
    exc12_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc12_N2"), 11.00, chamber)
    exc13_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc13_N2"), 11.90, chamber)
    exc14_N2 = Excitation(species, "N2", get_K_func(species, "N2", "exc14_N2"), 12.30, chamber)
# N
    exc1_N = Excitation(species, "N", get_K_func(species, "N", "exc1_N"), 3.20, chamber)
    exc2_N = Excitation(species, "N", get_K_func(species, "N", "exc2_N"), 4.00, chamber)
    # dion_N2 = Reaction(species_list, "N2", "N, N+", "N+", "e", get_K_func(species_list, "N2", "dion_N2"), 18.00, [1., 1., 1., 1.]) Hassoul 
# O2    
    exc1_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc1_O2"), 1.00, chamber)
    exc2_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc2_O2"), 1.50, chamber)
    exc3_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc3_O2"), 4.50, chamber)
    exc4_O2 = Excitation(species, "O2", get_K_func(species, "O2", "exc4_O2"), 7.10, chamber)
# O
    exc1_O = Excitation(species, "O", get_K_func(species, "O", "exc1_O"), 1.97, chamber)
    exc2_O = Excitation(species, "O", get_K_func(species, "O", "exc2_O"), 4.19, chamber)
    exc3_O = Excitation(species, "O", get_K_func(species, "O", "exc3_O"), 9.52, chamber)
    exc4_O = Excitation(species, "O", get_K_func(species, "O", "exc4_O"), 12, chamber)
    exc5_O = Excitation(species, "O", get_K_func(species, "O", "exc5_O"), 12, chamber)
    exc6_O = Excitation(species, "O", get_K_func(species, "O", "exc6_O"), 12, chamber)
    exc7_O = Excitation(species, "O", get_K_func(species, "O", "exc7_O"), 12, chamber)
    exc8_O = Excitation(species, "O", get_K_func(species, "O", "exc8_O"), 12, chamber)
    exc9_O = Excitation(species, "O", get_K_func(species, "O", "exc9_O"), 12, chamber)

# Ar
    exc1_Ar = Excitation(species, "Ar", get_K_func(species, "Ar", "exc_Ar"), 11.5, chamber)
    
#  █ ▄▀▄ █▄ █ █ ▄▀▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █ ▀▄▀ █ ▀█ █ ▄██ █▀█  █  █ ▀▄▀ █ ▀█
    ion_N = Ionisation(species, "N", "N+", get_K_func(species, "N", "ion_N"), 14.80, chamber)
    ion_N2 = Ionisation(species, "N2", "N2+", get_K_func(species, "N2", "ion_N2"), 15.60, chamber)
    ion_O2 = Ionisation(species, "O2", "O2+", get_K_func(species, "O2", "ion_O2"), 12.10, chamber)
    ion_O = Ionisation(species, "O", "O+", get_K_func(species,"O", "ion_O"), 13.60, chamber)
    ion_Ar = Ionisation(species, "Ar", "Ar+", get_K_func(species, "Ar", "ion_Ar"), 15.76, chamber)

#  ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀
#  █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██  # * complete
    ela_O2 = ElasticCollisionWithElectron(species, "O2", get_K_func(species, "O2", "ela_O2"), chamber)
    ela_N = ElasticCollisionWithElectron(species, "N", get_K_func(species, "N", "ela_N"), chamber)
    ela_O = ElasticCollisionWithElectron(species, "O", get_K_func(species, "O", "ela_O"), chamber)
    ela_N2 = ElasticCollisionWithElectron(species, "N2", get_K_func(species, "N2", "ela_N2"), chamber)
    ela_Ar = ElasticCollisionWithElectron(species, "Ar", get_K_func(species, "Ar", "ela_Ar"), chamber)


#  █ █ █ ██▄ █▀▄ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █ ▄▀▄ █     ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄▀ █ █▄█ █▀▄ █▀█  █  █ ▀▄▀ █ ▀█ █▀█ █▄▄   █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
    vib_exc_N2_list = VibrationalExcitation.from_concatenated_txt_file(species, "N2", "vib_exc", "EXCITATION", chamber)
    vib_exc_O2_list = VibrationalExcitation.from_concatenated_txt_file(species, "O2", "vib_exc", "EXCITATION", chamber)


#  █▀▄ ▄▀▄ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █ ▄▀▄ █     ██▀ ▀▄▀ ▄▀▀ █ ▀█▀ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▀▄ ▀▄▀  █  █▀█  █  █ ▀▄▀ █ ▀█ █▀█ █▄▄   █▄▄ █ █ ▀▄▄ █  █  █▀█  █  █ ▀▄▀ █ ▀█
    rot_exc_N2_list = RotationalExcitation.from_concatenated_txt_file(species, "N2", "rot_exc", "ROTATIONAL", chamber)
    rot_exc_O2_list = RotationalExcitation.from_concatenated_txt_file(species, "O2", "rot_exc", "ROTATIONAL", chamber)



#  █▀ █   █ █ ▀▄▀ ██▀ ▄▀▀   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀   ▄▀▄ █▄ █ █▀▄   ▀█▀ █▄█ █▀▄ ▄▀▄ █ █ ▄▀  █▄█   ▀█▀ █▄█ ██▀   ▄▀  █▀▄ █ █▀▄ ▄▀▀
#  █▀ █▄▄ ▀▄█ █ █ █▄▄ ▄██    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██   █▀█ █ ▀█ █▄▀    █  █ █ █▀▄ ▀▄▀ ▀▄█ ▀▄█ █ █    █  █ █ █▄▄   ▀▄█ █▀▄ █ █▄▀ ▄██
    out_flux = FluxToWallsAndThroughGrids(species, chamber)


#  ▄▀  ▄▀▄ ▄▀▀   █ █▄ █   █ ██▀ ▄▀▀ ▀█▀ █ ▄▀▄ █▄ █
#  ▀▄█ █▀█ ▄██   █ █ ▀█ ▀▄█ █▄▄ ▀▄▄  █  █ ▀▄▀ █ ▀█
    T_injection = 0.03 #à revoir
    gas_injection = GasInjection(species, injection_rates, T_injection, chamber)


#  █ █▄ █ ██▀ █   ▄▀▄ ▄▀▀ ▀█▀ █ ▄▀▀   ▄▀▀ ▄▀▄ █   █   █ ▄▀▀ █ ▄▀▄ █▄ █ ▄▀▀   █   █ █ ▀█▀ █▄█   █ ▄▀▄ █▄ █ ▄▀▀   ▄▀▄ ▄▀▀ ▄▀▀ ██▀ █   ██▀ █▀▄ ▄▀▄ ▀█▀ ██▀ █▀▄   ▀█▀ ▄▀▄   ▀█▀ █▄█ ██▀   █   █ ▄▀▄ █   █   ▄▀▀
#  █ █ ▀█ █▄▄ █▄▄ █▀█ ▄██  █  █ ▀▄▄   ▀▄▄ ▀▄▀ █▄▄ █▄▄ █ ▄██ █ ▀▄▀ █ ▀█ ▄██   ▀▄▀▄▀ █  █  █ █   █ ▀▄▀ █ ▀█ ▄██   █▀█ ▀▄▄ ▀▄▄ █▄▄ █▄▄ █▄▄ █▀▄ █▀█  █  █▄▄ █▄▀    █  ▀▄▀    █  █ █ █▄▄   ▀▄▀▄▀ █▀█ █▄▄ █▄▄ ▄██
    inelastic_collisions = InelasticCollision(species, chamber)

#  █▀▄ █ ▄▀▀ ▄▀▀ ▄▀▄ ▄▀▀ █ ▄▀▄ ▀█▀ █ ▄▀▄ █▄ █
#  █▄▀ █ ▄██ ▄██ ▀▄▀ ▀▄▄ █ █▀█  █  █ ▀▄▀ █ ▀█
    # Delta_E = 0.5 selon pifomètre d'Esteves ( = monoatomic_energy_excess )
    diss1_O2 = Dissociation(species, "O2", "O", get_K_func(species, "O2", "diss1_O2"), 6.12, 0.5, chamber)
    diss2_O2 = Dissociation(species, "O2", "O", get_K_func(species, "O2", "diss2_O2"), 8.40, 0.5, chamber)
    diss_N2 = Dissociation(species, "N2", "N", get_K_func(species, "N2", "diss_N2"), 9.76, 0.5, chamber)

#  ▀█▀ █▄█ ██▀ █▀▄ █▄ ▄█ █ ▄▀▀   █▀▄ █ █▀ █▀ █ █ ▄▀▀ █ ▄▀▄ █▄ █  
#   █  █ █ █▄▄ █▀▄ █ ▀ █ █ ▀▄▄   █▄▀ █ █▀ █▀ ▀▄█ ▄██ █ ▀▄▀ █ ▀█  
    #kappa = lambda T_i : 4.4e-5 * (e / k_B * T_i)**0.8  # noqa: E731
    kappa = dict()
    kappa["N"] = lambda T_i : 1.75*3.95e-4 * (e/k_B * T_i)**0.691 #3.95e-5 * (e/k_B * T_i)**0.691
    kappa["N2"] = lambda T_i : 1.75*2.06e-4 * (e/k_B * T_i)**0.754 #2.06e-5 * (e/k_B * T_i)**0.754
    kappa["O"] =  lambda T_i : 1.75*4.41e-4 * (e/k_B * T_i)**0.679 #4.41e-5 * (e/k_B * T_i)**0.679
    kappa["O2"] = lambda T_i : 1.75*1.66e-4 * (e/k_B * T_i)**0.798 #1.66e-5 * (e/k_B * T_i)**0.798
    kappa["Ar"] = lambda T_i : 1.75*3.59e-5 * (e/k_B * T_i)**0.72  #3.59e-6 * (e/k_B * T_i)**0.72
    #kappa = lambda T_i : 0.0
    th_diff = ThermicDiffusion(species, kappa, 0.03, chamber)

    #soit mettre le kappa en instance, soit faire une liste de kappa (mais faut l'associer à la bonne espèce)
    
    # Reaction list
    reaction_list = [
        exc1_N2, exc2_N2, exc3_N2, exc4_N2, exc5_N2, exc6_N2, exc7_N2, exc8_N2, exc9_N2, exc11_N2, exc12_N2, exc13_N2, exc14_N2, 
        exc1_N, exc2_N, exc1_O2, exc2_O2, exc3_O2, exc4_O2, 
        exc1_O, exc2_O, exc3_O, exc4_O, exc5_O, exc6_O, exc7_O, exc8_O, exc9_O,
        exc1_Ar, ion_Ar, ela_Ar,
        *vib_exc_N2_list, *vib_exc_O2_list, *rot_exc_N2_list, *rot_exc_O2_list,   # * is used to unpack lists (similar to *args in functions)
        ela_N, ela_N2, ela_O, ela_O2, 
        ion_N, ion_O2, ion_N2, ion_O,
        diss1_O2, diss2_O2, diss_N2,
        out_flux, gas_injection, inelastic_collisions, 
        th_diff
    ]

#  ██▀ █   ██▀ ▄▀▀ ▀█▀ █▀▄ ▄▀▄ █▄ █   █▄█ ██▀ ▄▀▄ ▀█▀ █ █▄ █ ▄▀    ██▄ ▀▄▀   ▀█▀ █▄█ ██▀   ▄▀▀ ▄▀▄ █ █    
#  █▄▄ █▄▄ █▄▄ ▀▄▄  █  █▀▄ ▀▄▀ █ ▀█   █ █ █▄▄ █▀█  █  █ █ ▀█ ▀▄█   █▄█  █     █  █ █ █▄▄   ▀▄▄ ▀▄▀ █ █▄▄  
    electron_heating = ElectronHeatingConstantRFPower(species, 1000, chamber)

    return species, initial_state, reaction_list, electron_heating
