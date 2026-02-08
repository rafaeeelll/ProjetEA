import numpy as np
from scipy.constants import pi, e, k as k_B, epsilon_0 as eps_0, c, m_e
from datetime import date as date_cls
from datetime import datetime
from functools import lru_cache
from pathlib import Path
# import numpy as np

from global_model_package.reactions import (Excitation, Ionisation, Dissociation, 
                VibrationalExcitation, RotationalExcitation,
                ThermicDiffusion, InelasticCollision, ElasticCollisionWithElectron, 
                FluxToWallsAndThroughGrids, GasInjection,
                ElectronHeatingConstantRFPower, ElectronHeatingConstantAbsorbedPower
            )

from global_model_package.specie import Species, Specie
from global_model_package.constant_rate_calculation import get_K_func, ReactionRateConstant

ReactionRateConstant.CROSS_SECTIONS_PATH = "../../../cross_sections"

DEFAULT_MSIS_DATE = datetime(2020, 1, 1, 12, 0, 0)
DEFAULT_MSIS_LAT = 0.0
DEFAULT_MSIS_LON = 0.0
DEFAULT_MSIS_F107 = None
DEFAULT_MSIS_F107A = None
DEFAULT_MSIS_AP = None

SPACE_WEATHER_PATH = Path(__file__).resolve().parents[3].joinpath("data", "space_weather.txt")


def _parse_space_weather(path: Path) -> dict[date_cls, dict[str, object]]:
    records: dict[date_cls, dict[str, object]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 28:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
            except ValueError:
                continue
            day_key = date_cls(year, month, day)
            try:
                ap_3h = [int(p) for p in parts[15:23]]
                ap_daily = float(parts[23])
                f107_obs = float(parts[25])
                f107_adj = float(parts[26])
            except ValueError:
                continue
            records[day_key] = {
                "ap_3h": ap_3h,
                "ap_daily": ap_daily,
                "f107_obs": f107_obs,
                "f107_adj": f107_adj,
            }
    return records


def _compute_f107a(records: dict[date_cls, dict[str, object]]) -> dict[date_cls, float]:
    if not records:
        return {}
    dates_sorted = sorted(records.keys())
    f107a_map: dict[date_cls, float] = {}
    for idx, day_key in enumerate(dates_sorted):
        window_start = max(0, idx - 80)
        window_dates = dates_sorted[window_start : idx + 1]
        vals = []
        for d in window_dates:
            v = float(records[d]["f107_obs"])
            if v >= 0:
                vals.append(v)
        if vals:
            f107a_map[day_key] = float(np.mean(vals))
        else:
            f107a_map[day_key] = float(records[day_key]["f107_adj"])
    return f107a_map


def _nearest_available_date(target: date_cls, records: dict[date_cls, dict[str, object]]) -> date_cls:
    if target in records:
        return target
    if not records:
        raise ValueError("space_weather.txt has no data rows.")
    dates_sorted = sorted(records.keys())
    if target < dates_sorted[0]:
        return dates_sorted[0]
    if target > dates_sorted[-1]:
        return dates_sorted[-1]
    for d in reversed(dates_sorted):
        if d <= target:
            return d
    return dates_sorted[0]


def _space_weather_params(
    dt: datetime,
    records: dict[date_cls, dict[str, object]],
    f107a_map: dict[date_cls, float],
) -> tuple[float, float, float]:
    day_key = _nearest_available_date(dt.date(), records)
    rec = records[day_key]
    f107_obs = float(rec["f107_obs"])
    f107_adj = float(rec["f107_adj"])
    f107 = f107_obs if f107_obs >= 0 else f107_adj
    f107a = f107a_map.get(day_key, f107_adj)
    ap_daily = float(rec["ap_daily"])
    if ap_daily < 0:
        ap_3h = rec.get("ap_3h", [])
        ap_vals = [float(v) for v in ap_3h if float(v) >= 0]
        if ap_vals:
            ap_daily = float(np.mean(ap_vals))
    return f107, f107a, ap_daily


@lru_cache(maxsize=1)
def _load_space_weather() -> tuple[dict[date_cls, dict[str, object]], dict[date_cls, float]]:
    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)
    return records, f107a_map


def _msise_atmosphere(altitude_km, lat, lon, date, f107, f107a, ap):
    from nrlmsise00 import msise_model  # type: ignore

    dens, temp = msise_model(date, altitude_km, lat, lon, f107a, f107, ap)
    dens = np.array(dens, dtype=float)
    temp = np.array(temp, dtype=float)

    # NRLMSISE outputs number densities in cm^-3; convert to m^-3
    n_N2 = dens[2] * 1e6
    n_O2 = dens[3] * 1e6
    n_O = dens[1] * 1e6
    n_N = dens[7] * 1e6
    n_total = n_N2 + n_O2 + n_O + n_N

    T_K = float(temp[1])
    return {
        "N2": float(n_N2),
        "N": float(n_N),
        "O2": float(n_O2),
        "O": float(n_O),
        "T_K": T_K,
        "T_eV": float(k_B * T_K / e),
        "source": "msise",
    }


def get_neutral_atmosphere(
    altitude_km,
    lat=DEFAULT_MSIS_LAT,
    lon=DEFAULT_MSIS_LON,
    date=DEFAULT_MSIS_DATE,
    f107=DEFAULT_MSIS_F107,
    f107a=DEFAULT_MSIS_F107A,
    ap=DEFAULT_MSIS_AP,
):
    if f107 is None or f107a is None or ap is None:
        records, f107a_map = _load_space_weather()
        if records:
            f107, f107a, ap = _space_weather_params(date, records, f107a_map)
        else:
            f107 = 150.0
            f107a = 150.0
            ap = 4.0
    msise = _msise_atmosphere(altitude_km, lat, lon, date, f107, f107a, ap)
    n_total = msise["N2"] + msise["N"] + msise["O2"] + msise["O"]
    msise["pressure_pa"] = float(n_total * k_B * msise["T_K"])
    return msise


def get_species_and_reactions(
    chamber,
    altitude,
    argon_injection_rate,
    lat=DEFAULT_MSIS_LAT,
    lon=DEFAULT_MSIS_LON,
    date=DEFAULT_MSIS_DATE,
    ion_seed=1e8,
    electron_seed=2.1e12,
    compression_rate=500,
    collection_rate=1,
    atm=None,
):

    species = Species([Specie("e", m_e, -e, 0, 3/2), Specie("Ar", 6.63e-26, 0, 1, 3/2), Specie("Ar+", 6.63e-26, e, 1, 3/2), Specie("N2", 4.65e-26, 0, 2, 5/2), Specie("N", 2.33e-26, 0, 1, 3/2), Specie("N2+", 4.65e-26, e, 2, 5/2), Specie("N+", 2.33e-26, e, 1, 3/2), Specie("O2+", 5.31e-26, e, 2, 5/2), Specie("O2", 5.31e-26, 0, 2, 5/2), Specie("O", 2.67e-26, 0, 1, 3/2), Specie("O+", 2.67e-26, e, 1, 3/2)])

    if atm is None:
        atm = get_neutral_atmosphere(altitude, lat=lat, lon=lon, date=date)

    initial_state_dict = {
        "e": electron_seed,
        "Ar" : 1e14,
        "Ar+" : ion_seed,
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
    print(initial_state_dict)

    # Apply compression to neutrals only; compressing electron seeds makes
    # eps_p extremely negative and breaks RF power absorption (NaNs).
    initial_densities = []
    for sp in species.species:
        base_density = initial_state_dict[sp.name]
        if sp.charge == 0 and sp.name != "e":
            initial_densities.append(compression_rate * base_density)
        else:
            initial_densities.append(base_density)
    initial_state = initial_densities + [
        initial_state_dict["T_e"],
        initial_state_dict["T_mono"],
        initial_state_dict["T_diato"],
    ]
    
    injection_rates = np.zeros(species.nb)
    injection_rates[species.get_specie_by_name("N2").index] = collection_rate * atm["N2"] *compression_rate
    injection_rates[species.get_specie_by_name("N").index] = collection_rate * atm["N"] *compression_rate
    injection_rates[species.get_specie_by_name("O2").index] = collection_rate * atm["O2"]*compression_rate
    injection_rates[species.get_specie_by_name("O").index] = collection_rate * atm["O"]*compression_rate
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

    print(injection_rates)

    return species, initial_state, reaction_list, electron_heating
