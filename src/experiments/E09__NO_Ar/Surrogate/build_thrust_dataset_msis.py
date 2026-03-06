from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


E09_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")

DATE = datetime(2020, 1, 1, 12, 0, 0)
N_POINTS = 200
RANDOM_SEED = 42

# Global envelope
ALTITUDE_BOUNDS_KM = (160.0, 220.0)
AREA_BOUNDS_M2 = (5e-2, 0.8)
# Focused envelope around the expected feasible/optimal region
ALTITUDE_FOCUS_BOUNDS_KM = (170.0, 190.0)
AREA_FOCUS_BOUNDS_M2 = (0.12, 0.32)
FOCUS_FRACTION = 0.8
INCLINATION_BOUNDS_DEG = (0.0, 98.0)
RAAN_BOUNDS_DEG = (-180.0, 180.0)
EARTH_RADIUS_M = 6371e3
EARTH_MU = 3.986004418e14
EARTH_OMEGA = 7.2921159e-5


if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

from msis_densities import _compute_f107a, _parse_space_weather, _space_weather_params, SPACE_WEATHER_PATH


def _orbital_point(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    theta_rad: float,
    date: datetime,
) -> tuple[datetime, float, float, float]:
    radius = EARTH_RADIUS_M + altitude_km * 1000.0
    mean_motion = np.sqrt(EARTH_MU / radius**3)
    orbital_speed_m_s = np.sqrt(EARTH_MU / radius)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)

    t = theta_rad / mean_motion
    dt = date + timedelta(seconds=float(t))

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


def _msis_state(altitude_km: float, dt: datetime, lat_deg: float, lon_deg: float, f107: float, f107a: float, ap: float):
    from nrlmsise00 import msise_model

    dens, temp = msise_model(dt, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
    dens = np.array(dens, dtype=float)
    return {
        "N2_m3": float(dens[2]) * 1e6,
        "O2_m3": float(dens[3]) * 1e6,
        "O_m3": float(dens[1]) * 1e6,
        "N_m3": float(dens[7]) * 1e6,
        "T_K": float(temp[1]),
    }


def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    u = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = (perm + rng.random(n)) / n
    return u


def _interp_bounds(u_scalar: float, bounds: tuple[float, float]) -> float:
    return float(bounds[0] + float(u_scalar) * (bounds[1] - bounds[0]))


def _interp_log_bounds(u_scalar: float, bounds: tuple[float, float]) -> float:
    log_low = np.log10(bounds[0])
    log_high = np.log10(bounds[1])
    return float(10 ** (log_low + float(u_scalar) * (log_high - log_low)))


def main() -> None:
    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)
    f1070, f107a0, ap0 = _space_weather_params(DATE, records, f107a_map)

    rng = np.random.default_rng(RANDOM_SEED)
    u = _lhs_unit(N_POINTS, 5, rng)
    n_focus = int(round(FOCUS_FRACTION * N_POINTS))

    samples = []
    for i in range(N_POINTS):
        in_focus = i < n_focus
        altitude_bounds = ALTITUDE_FOCUS_BOUNDS_KM if in_focus else ALTITUDE_BOUNDS_KM
        area_bounds = AREA_FOCUS_BOUNDS_M2 if in_focus else AREA_BOUNDS_M2

        altitude_km = _interp_bounds(u[i, 0], altitude_bounds)
        inclination_deg = _interp_bounds(u[i, 1], INCLINATION_BOUNDS_DEG)
        raan_deg = _interp_bounds(u[i, 2], RAAN_BOUNDS_DEG)
        theta_rad = 2.0 * np.pi * u[i, 3]
        area_m2 = _interp_log_bounds(u[i, 4], area_bounds)
        dt, lat_deg, lon_deg, orbital_speed_m_s = _orbital_point(
            float(altitude_km),
            float(inclination_deg),
            float(raan_deg),
            float(theta_rad),
            DATE,
        )
        f107, f107a, ap = _space_weather_params(dt, records, f107a_map)
        st = _msis_state(float(altitude_km), dt, lat_deg, lon_deg, f107, f107a, ap)

        samples.append(
            {
                "N2_m3": float(st["N2_m3"]),
                "O2_m3": float(st["O2_m3"]),
                "O_m3": float(st["O_m3"]),
                "N_m3": float(st["N_m3"]),
                "T_K": float(st["T_K"]),
                "intake_area_m2": float(area_m2),
                "altitude_km": float(altitude_km),
                "orbital_speed_m_s": float(orbital_speed_m_s),
                "inclination_deg": float(inclination_deg),
                "raan_deg": float(raan_deg),
                "theta_rad": float(theta_rad),
                "lat_deg": float(lat_deg),
                "lon_deg": float(lon_deg),
                "argon_injection_rate": 0.0,
                "date": dt.isoformat(),
                "f107": float(f107),
                "f107a": float(f107a),
                "ap": float(ap),
                "total_thrust_N": None,
                "needs_simulation": True,
                "sampling_zone": "focus" if in_focus else "global",
            }
        )

    n2_vals = np.array([s["N2_m3"] for s in samples], dtype=float)
    o2_vals = np.array([s["O2_m3"] for s in samples], dtype=float)
    o_vals = np.array([s["O_m3"] for s in samples], dtype=float)
    n_vals = np.array([s["N_m3"] for s in samples], dtype=float)
    t_vals = np.array([s["T_K"] for s in samples], dtype=float)

    payload = {
        "metadata": {
            "description": "Seed dataset sampled from coherent MSIS orbit states (no independent density sampling).",
            "date": DATE.isoformat(),
            "f107_at_ref_date": float(f1070),
            "f107a_at_ref_date": float(f107a0),
            "ap_at_ref_date": float(ap0),
            "n_points": int(N_POINTS),
            "n_focus_points": int(n_focus),
            "random_seed": int(RANDOM_SEED),
            "altitude_bounds_km": list(ALTITUDE_BOUNDS_KM),
            "altitude_focus_bounds_km": list(ALTITUDE_FOCUS_BOUNDS_KM),
            "inclination_bounds_deg": list(INCLINATION_BOUNDS_DEG),
            "raan_bounds_deg": list(RAAN_BOUNDS_DEG),
            "area_bounds_m2": list(AREA_BOUNDS_M2),
            "area_focus_bounds_m2": list(AREA_FOCUS_BOUNDS_M2),
            "focus_fraction": float(FOCUS_FRACTION),
            "argon_injection_rate_bounds": [0.0, 0.0],
            "density_bounds_m3": {
                "N2_m3": [float(np.min(n2_vals)), float(np.max(n2_vals))],
                "O2_m3": [float(np.min(o2_vals)), float(np.max(o2_vals))],
                "O_m3": [float(np.min(o_vals)), float(np.max(o_vals))],
                "N_m3": [float(np.min(n_vals)), float(np.max(n_vals))],
            },
            "temperature_bounds_K": [float(np.min(t_vals)), float(np.max(t_vals))],
        },
        "samples": samples,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {len(samples)} seed samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
