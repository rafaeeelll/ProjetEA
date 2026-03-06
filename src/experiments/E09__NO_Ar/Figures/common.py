from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.constants import e, k as k_B, pi


E09_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = PROJECT_ROOT.joinpath("figures", "E09")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

try:
    import global_model_package  # noqa: F401
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parents[3].joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.chamber_caracteristics import Chamber  # type: ignore
from global_model_package.model import GlobalModel  # type: ignore
from global_model_package.reactions import ElectronHeatingConstantRFPower  # type: ignore

from Drag.drag_model import (
    A_BODY_M2_DEFAULT,
    CD_BODY_DEFAULT,
    collection_efficiency,
    drag_body_fmf,
    drag_intake_fmf,
    mass_density_from_number_densities,
)
from msis_densities import (
    SPACE_WEATHER_PATH,
    _compute_f107a,
    _parse_space_weather,
    _space_weather_params,
    get_msis_neutral_atmosphere,
)
from reaction_set_N_et_O import OUTLET_AREA_M2, compute_beta, get_species_and_reactions


EARTH_RADIUS_M = 6371e3
EARTH_MU = 3.986004418e14
EARTH_OMEGA = 7.2921159e-5
M_AR = 6.63e-26
SPECIES_MASS = {
    "N2": 4.65e-26,
    "O2": 5.31e-26,
    "O": 2.67e-26,
    "N": 2.33e-26,
}

CHAMBER_CONFIG = {
    "R": 6e-2,
    "L": 10e-2,
    "V_grid": 1000,
    "beta_i": 0.7,
    "beta_g": 0.3,
    "omega": 13.56e6 * 2 * pi,
    "N": 5,
    "R_coil": 2,
}


@dataclass
class Case:
    altitude_km: float
    area_m2: float
    argon_rate: float = 0.0
    inclination_deg: float = 20.0
    raan_deg: float = -13.0
    theta_rad: float = 0.0
    date_ref: datetime = datetime(2020, 1, 1, 12, 0, 0)


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def weather_records():
    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)
    return records, f107a_map


def orbit_state(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    theta_rad: float,
    date_ref: datetime,
) -> tuple[datetime, float, float, float]:
    radius = EARTH_RADIUS_M + altitude_km * 1000.0
    mean_motion = np.sqrt(EARTH_MU / radius**3)
    orbital_speed_m_s = np.sqrt(EARTH_MU / radius)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    t = theta_rad / mean_motion
    dt = date_ref + timedelta(seconds=float(t))

    x_orb = radius * np.cos(theta_rad)
    y_orb = radius * np.sin(theta_rad)
    x_inc = x_orb
    y_inc = y_orb * np.cos(inc)
    z_inc = y_orb * np.sin(inc)

    x_eci = x_inc * np.cos(raan) - y_inc * np.sin(raan)
    y_eci = x_inc * np.sin(raan) + y_inc * np.cos(raan)
    z_eci = z_inc

    cos_e = np.cos(EARTH_OMEGA * t)
    sin_e = np.sin(EARTH_OMEGA * t)
    x_ecef = x_eci * cos_e + y_eci * sin_e
    y_ecef = -x_eci * sin_e + y_eci * cos_e
    z_ecef = z_eci
    lat_deg = float(np.degrees(np.arcsin(z_ecef / radius)))
    lon_deg = float(np.degrees(np.arctan2(y_ecef, x_ecef)))
    return dt, lat_deg, lon_deg, float(orbital_speed_m_s)


def msis_state_from_case(case: Case, records: dict | None = None, f107a_map: dict | None = None) -> dict:
    if records is None or f107a_map is None:
        records, f107a_map = weather_records()
    dt, lat_deg, lon_deg, orbital_speed = orbit_state(
        altitude_km=case.altitude_km,
        inclination_deg=case.inclination_deg,
        raan_deg=case.raan_deg,
        theta_rad=case.theta_rad,
        date_ref=case.date_ref,
    )
    f107, f107a, ap = _space_weather_params(dt, records, f107a_map)
    atm = get_msis_neutral_atmosphere(
        altitude_km=case.altitude_km,
        lat=lat_deg,
        lon=lon_deg,
        date=dt,
        f107=f107,
        f107a=f107a,
        ap=ap,
    )
    return {
        "N2_m3": float(atm["N2"]),
        "O2_m3": float(atm["O2"]),
        "O_m3": float(atm["O"]),
        "N_m3": float(atm["N"]),
        "T_K": float(atm["T_K"]),
        "T_eV": float(atm["T_eV"]),
        "orbital_speed_m_s": float(orbital_speed),
        "lat_deg": float(lat_deg),
        "lon_deg": float(lon_deg),
        "f107": float(f107),
        "f107a": float(f107a),
        "ap": float(ap),
        "date": dt.isoformat(),
    }


def build_sample_from_case(case: Case, records: dict | None = None, f107a_map: dict | None = None) -> dict:
    st = msis_state_from_case(case, records=records, f107a_map=f107a_map)
    eta_eff = collection_efficiency(case.area_m2)
    return {
        "N2_m3": st["N2_m3"],
        "O2_m3": st["O2_m3"],
        "O_m3": st["O_m3"],
        "N_m3": st["N_m3"],
        "T_K": st["T_K"],
        "intake_area_m2": float(case.area_m2),
        "argon_injection_rate": float(case.argon_rate),
        "altitude_km": float(case.altitude_km),
        "orbital_speed_m_s": float(st["orbital_speed_m_s"]),
        "inclination_deg": float(case.inclination_deg),
        "raan_deg": float(case.raan_deg),
        "theta_rad": float(case.theta_rad),
        "lat_deg": float(st["lat_deg"]),
        "lon_deg": float(st["lon_deg"]),
        "date": st["date"],
        "f107": float(st["f107"]),
        "f107a": float(st["f107a"]),
        "ap": float(st["ap"]),
        "eta_collection": float(eta_eff),
    }


def mass_density_from_sample(sample: dict) -> float:
    return mass_density_from_number_densities(
        n2_m3=float(sample["N2_m3"]),
        o2_m3=float(sample["O2_m3"]),
        o_m3=float(sample["O_m3"]),
        n_m3=float(sample["N_m3"]),
    )


def drag_components(
    sample: dict,
    cd_body: float = CD_BODY_DEFAULT,
    body_area_m2: float = A_BODY_M2_DEFAULT,
    lateral_area_m2: float = 0.0,
    cd_lateral: float = 2.2,
) -> dict:
    rho = mass_density_from_sample(sample)
    u = float(sample["orbital_speed_m_s"])
    a_intake = float(sample["intake_area_m2"])
    eta_eff = collection_efficiency(a_intake)
    d_intake = drag_intake_fmf(rho=rho, speed_m_s=u, intake_area_m2=a_intake)
    d_body = drag_body_fmf(rho=rho, speed_m_s=u, cd_body=cd_body, a_body_m2=body_area_m2)
    d_lateral = 0.5 * rho * u * u * float(cd_lateral) * float(lateral_area_m2)
    return {
        "rho": float(rho),
        "eta_collection": float(eta_eff),
        "drag_intake_N": float(d_intake),
        "drag_body_N": float(d_body),
        "drag_lateral_N": float(d_lateral),
        "drag_total_N": float(d_intake + d_body + d_lateral),
    }


def beta_by_species(orbital_speed_m_s: float, t_wall_k: float, a_intake_m2: float, eta_collection: float) -> dict:
    out = {}
    for name, mass in SPECIES_MASS.items():
        out[name] = float(
            compute_beta(
                u_orbital=orbital_speed_m_s,
                T_wall_K=t_wall_k,
                A_intake=a_intake_m2,
                A_outlet=OUTLET_AREA_M2,
                eta_c=eta_collection,
                m_species=mass,
            )
        )
    return out


def solve_plasma_sample(
    sample: dict,
    power_rf_w: float = 1000.0,
    fast_mode: bool = True,
    t0: float = 0.0,
    tf: float = 1e-2,
    simulation_name: str = "figure_case",
) -> dict:
    chamber = Chamber(CHAMBER_CONFIG)
    atm = {
        "N2": float(sample["N2_m3"]),
        "N": float(sample["N_m3"]),
        "O2": float(sample["O2_m3"]),
        "O": float(sample["O_m3"]),
        "T_K": float(sample["T_K"]),
        "T_eV": float(k_B * float(sample["T_K"]) / e),
        "source": "figure_helper",
    }
    eta_eff = collection_efficiency(float(sample["intake_area_m2"]))
    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber=chamber,
        altitude=float(sample.get("altitude_km", 200.0)),
        argon_injection_rate=float(sample.get("argon_injection_rate", 0.0)),
        ion_seed=1e10,
        electron_seed=1e12,
        A_intake=float(sample["intake_area_m2"]),
        eta_collection=float(eta_eff),
        orbital_speed=float(sample["orbital_speed_m_s"]),
        atm=atm,
    )
    electron_heating = ElectronHeatingConstantRFPower(species, float(power_rf_w), chamber)
    log_dir = PROJECT_ROOT.joinpath("outputs", "e09_figure_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name=simulation_name,
        log_folder_path=log_dir,
        fast=fast_mode,
    )
    sol = model.solve(t0, tf, initial_state)
    if not getattr(sol, "success", False):
        raise RuntimeError(
            f"Plasma solve failed for {simulation_name}: "
            f"status={getattr(sol, 'status', None)} "
            f"message={getattr(sol, 'message', 'unknown')}"
        )
    states = np.asarray(sol.y, dtype=float).T
    thrust_series = np.array([float(model.total_thrust(state)) for state in states], dtype=float)
    thrust_final = float(thrust_series[-1])
    return {
        "species": species,
        "model": model,
        "solution": sol,
        "states": states,
        "thrust_series": thrust_series,
        "thrust_final_N": thrust_final,
        "eta_collection_eff": float(eta_eff),
    }


def mdot_kg_s(sample: dict) -> float:
    eta_eff = collection_efficiency(float(sample["intake_area_m2"]))
    capture = eta_eff * float(sample["orbital_speed_m_s"]) * float(sample["intake_area_m2"])
    mdot_air = capture * mass_density_from_sample(sample)
    mdot_argon = float(sample.get("argon_injection_rate", 0.0)) * M_AR
    return float(mdot_air + mdot_argon)


def isp_s(thrust_n: float, mdot: float) -> float:
    if mdot <= 0.0:
        return float("inf")
    return float(thrust_n / (mdot * 9.81))


def ion_thrust_breakdown(model: GlobalModel, species, state: np.ndarray) -> dict:
    n_g = model.n_g_tot(state)
    te = float(state[species.nb])
    out: dict[str, float] = {}
    for sp in species.species[1:]:
        if sp.charge == 0:
            continue
        contrib = (
            model.chamber.gamma_ion(state[sp.index], te, sp.mass)
            * model.chamber.h_L(n_g)
            * sp.mass
            * model.chamber.v_beam(sp.mass, sp.charge)
            * model.chamber.beta_i
            * model.chamber.S_gridded_wall
        )
        out[sp.name] = float(contrib)
    out["neutral"] = float(model.total_neutral_thrust(state))
    return out


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
