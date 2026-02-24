from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from surrogate_config import (
    ARGON_LOG_EPS,
    CD_FRONT,
    ETA_COLLECTION,
    MDOT_LOG_EPS,
)


FEATURE_NAMES = [
    "log10_N2_m3",
    "log10_O2_m3",
    "log10_O_m3",
    "log10_N_m3",
    "T_K",
    "log10_argon_eps",
    "log10_mdot_in_kg_s",
    "log10_intake_area_m2",
]


def safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def rho_from_species_densities(n2_m3: float, o2_m3: float, o_m3: float, n_m3: float) -> float:
    return n2_m3 * 4.65e-26 + o2_m3 * 5.31e-26 + o_m3 * 2.67e-26 + n_m3 * 2.33e-26


def orbital_speed_from_altitude_km(altitude_km: float) -> float:
    mu_earth = 3.986004418e14
    r_earth = 6371e3
    radius = r_earth + altitude_km * 1000.0
    return float(np.sqrt(mu_earth / radius))


def mdot_in_kg_s(rho_kg_m3: float, speed_m_s: float, intake_area_m2: float, eta_c: float = ETA_COLLECTION) -> float:
    return rho_kg_m3 * speed_m_s * intake_area_m2 * eta_c


def drag_front_force(rho_kg_m3: np.ndarray, speed_m_s: float, intake_area_m2: float) -> np.ndarray:
    return 0.5 * rho_kg_m3 * speed_m_s**2 * CD_FRONT * intake_area_m2


def make_feature_row(
    n2_m3: float,
    o2_m3: float,
    o_m3: float,
    n_m3: float,
    t_k: float,
    argon_injection_rate: float,
    intake_area_m2: float,
    rho_kg_m3: float,
    speed_m_s: float,
) -> list[float]:
    mdot = mdot_in_kg_s(rho_kg_m3, speed_m_s, intake_area_m2)
    return [
        float(safe_log10(np.array([n2_m3]))[0]),
        float(safe_log10(np.array([o2_m3]))[0]),
        float(safe_log10(np.array([o_m3]))[0]),
        float(safe_log10(np.array([n_m3]))[0]),
        float(t_k),
        float(safe_log10(np.array([argon_injection_rate + ARGON_LOG_EPS]))[0]),
        float(safe_log10(np.array([mdot + MDOT_LOG_EPS]))[0]),
        float(safe_log10(np.array([intake_area_m2]))[0]),
    ]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
