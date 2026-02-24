from __future__ import annotations

import numpy as np

from global_model_package.reactions import GasInjection
from surrogate_config import ETA_COLLECTION

from reaction_set_N_et_O import (  # type: ignore
    get_neutral_atmosphere as _base_get_neutral_atmosphere,
    get_species_and_reactions as _base_get_species_and_reactions,
)


def get_neutral_atmosphere(*args, **kwargs):
    return _base_get_neutral_atmosphere(*args, **kwargs)


def _rho_from_atm(atm: dict[str, float]) -> float:
    return (
        atm["N2"] * 4.65e-26
        + atm["O2"] * 5.31e-26
        + atm["O"] * 2.67e-26
        + atm["N"] * 2.33e-26
    )


def get_species_and_reactions(
    chamber,
    altitude,
    argon_injection_rate,
    *,
    intake_area_m2: float,
    onset_speed_m_s: float,
    eta_collection: float = ETA_COLLECTION,
    lat=0.0,
    lon=0.0,
    date=None,
    ion_seed=1e8,
    electron_seed=2.1e12,
    atm=None,
):
    if atm is None:
        atm = _base_get_neutral_atmosphere(altitude, lat=lat, lon=lon, date=date)

    species, initial_state, reactions_list, electron_heating = _base_get_species_and_reactions(
        chamber=chamber,
        altitude=altitude,
        argon_injection_rate=argon_injection_rate,
        lat=lat,
        lon=lon,
        date=date,
        ion_seed=ion_seed,
        electron_seed=electron_seed,
        compression_rate=1.0,
        collection_rate=0.0,
        atm=atm,
    )

    n_total = max(atm["N2"] + atm["O2"] + atm["O"] + atm["N"], 1.0)
    rho = _rho_from_atm(atm)
    mdot_in = rho * onset_speed_m_s * intake_area_m2 * eta_collection

    inj = np.zeros(species.nb, dtype=float)
    for name in ("N2", "O2", "O", "N"):
        frac = atm[name] / n_total
        sp = species.get_specie_by_name(name)
        inj[sp.index] = mdot_in * frac / sp.mass

    inj[species.get_specie_by_name("Ar").index] = float(argon_injection_rate)

    gas_reaction = None
    for reaction in reactions_list:
        if isinstance(reaction, GasInjection):
            gas_reaction = reaction
            break
    if gas_reaction is None:
        raise RuntimeError("GasInjection reaction not found in reaction set.")
    gas_reaction.injection_rates = inj

    return species, initial_state, reactions_list, electron_heating
