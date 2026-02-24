from __future__ import annotations

import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.constants import pi
from scipy.stats import norm

from surrogate_config import BO_LOG_PATH, E09_DIR, ETA_COLLECTION, GP_DATASET_PATH, GP_MODEL_PATH
from gp_utils import (
    drag_front_force,
    make_feature_row,
    orbital_speed_from_altitude_km,
    rho_from_species_densities,
    read_json,
    write_json,
)

if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

from msis_densities import _compute_f107a, _parse_space_weather, _space_weather_params, SPACE_WEATHER_PATH
from reaction_set_N_et_O import get_neutral_atmosphere
from reaction_set_surrogate import get_species_and_reactions

try:
    import global_model_package  # noqa: F401
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parents[2].joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.chamber_caracteristics import Chamber
from global_model_package.model import GlobalModel
from global_model_package.reactions import ElectronHeatingConstantRFPower


# Scenario and BO settings
ALTITUDE_KM = 190.0
INCLINATION_DEG = 51.6
RAAN_DEG = 0.0
ORBIT_POINTS = 12
DATE = datetime(2020, 1, 1, 12, 0, 0)
ARGON_INJECTION_RATE = 0.0
POWER_RF_W = 1000.0
FAST_MODE = True
ION_SEED = 1e10
ELECTRON_SEED = 1e12

A_CANDIDATES_M2 = np.logspace(-2, 0, 9)
XI = 1e-8
N_ITER = 1


def _load_gp():
    with GP_MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    return model["gp"], model["x_mean"], model["x_std"]


def _orbital_speed(altitude_km: float) -> float:
    return orbital_speed_from_altitude_km(altitude_km)


def _orbit_points(altitude_km: float, inclination_deg: float, raan_deg: float, orbit_points: int, date: datetime):
    mu_earth = 3.986004418e14
    r_earth = 6371e3
    omega_earth = 7.2921159e-5
    radius = r_earth + altitude_km * 1000.0
    mean_motion = np.sqrt(mu_earth / radius**3)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    thetas = np.linspace(0.0, 2.0 * np.pi, orbit_points, endpoint=False)
    out = []
    for theta in thetas:
        t = theta / mean_motion
        dt = date + timedelta(seconds=float(t))
        x_orb = radius * np.cos(theta)
        y_orb = radius * np.sin(theta)
        x_inc = x_orb
        y_inc = y_orb * np.cos(inc)
        z_inc = y_orb * np.sin(inc)
        x_eci = x_inc * np.cos(raan) - y_inc * np.sin(raan)
        y_eci = x_inc * np.sin(raan) + y_inc * np.cos(raan)
        z_eci = z_inc
        cos_e = np.cos(omega_earth * t)
        sin_e = np.sin(omega_earth * t)
        x_ecef = x_eci * cos_e + y_eci * sin_e
        y_ecef = -x_eci * sin_e + y_eci * cos_e
        z_ecef = z_eci
        lat = float(np.degrees(np.arcsin(z_ecef / radius)))
        lon = float(np.degrees(np.arctan2(y_ecef, x_ecef)))
        out.append({"theta": float(theta), "t": float(t), "date": dt, "lat": lat, "lon": lon})
    return out


def _msis_with_rho(alt_km: float, lat: float, lon: float, dt: datetime, f107: float, f107a: float, ap: float):
    from nrlmsise00 import msise_model  # type: ignore

    dens, temp = msise_model(dt, alt_km, lat, lon, f107a, f107, ap)
    dens = np.array(dens, dtype=float)
    n2 = float(dens[2]) * 1e6
    o2 = float(dens[3]) * 1e6
    o = float(dens[1]) * 1e6
    n = float(dens[7]) * 1e6
    t_k = float(temp[1])
    rho = n2 * 4.65e-26 + o2 * 5.31e-26 + o * 2.67e-26 + n * 2.33e-26
    return n2, o2, o, n, t_k, rho


def _evaluate_orbit_score(area_m2: float, orbit_pts: list[dict], gp, x_mean: np.ndarray, x_std: np.ndarray, f107: float, f107a: float, ap: float):
    speed = _orbital_speed(ALTITUDE_KM)
    feats = []
    rho_arr = []
    for p in orbit_pts:
        n2, o2, o, n, t_k, rho = _msis_with_rho(ALTITUDE_KM, p["lat"], p["lon"], p["date"], f107, f107a, ap)
        feats.append(make_feature_row(n2, o2, o, n, t_k, ARGON_INJECTION_RATE, area_m2, rho, speed))
        rho_arr.append(rho)
    x = np.array(feats, dtype=float)
    xs = (x - x_mean) / x_std
    mu_t, std_t = gp.predict(xs, return_std=True)
    drag = drag_front_force(np.array(rho_arr, dtype=float), speed, area_m2)
    mu_margin = mu_t - drag
    idx_star = int(np.argmin(mu_margin))
    g_mu = float(mu_margin[idx_star])
    g_sigma = float(std_t[idx_star])
    return g_mu, g_sigma, idx_star


def _ei(mu: float, sigma: float, best: float, xi: float) -> float:
    if sigma <= 1e-14:
        return 0.0
    imp = mu - best - xi
    z = imp / sigma
    return float(imp * norm.cdf(z) + sigma * norm.pdf(z))


def _evaluate_true_thrust(orbit_point: dict, area_m2: float, f107: float, f107a: float, ap: float) -> float:
    config = {
        "R": np.sqrt(area_m2 / pi),
        "L": 10e-2,
        "V_grid": 1000,
        "beta_i": 0.7,
        "beta_g": 0.3,
        "omega": 13.56e6 * 2 * pi,
        "N": 5,
        "R_coil": 2,
    }
    chamber = Chamber(config)
    atm = get_neutral_atmosphere(
        ALTITUDE_KM,
        lat=orbit_point["lat"],
        lon=orbit_point["lon"],
        date=orbit_point["date"],
        f107=f107,
        f107a=f107a,
        ap=ap,
    )
    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber,
        ALTITUDE_KM,
        argon_injection_rate=ARGON_INJECTION_RATE,
        intake_area_m2=area_m2,
        onset_speed_m_s=_orbital_speed(ALTITUDE_KM),
        eta_collection=ETA_COLLECTION,
        lat=orbit_point["lat"],
        lon=orbit_point["lon"],
        date=orbit_point["date"],
        ion_seed=ION_SEED,
        electron_seed=ELECTRON_SEED,
        atm=atm,
    )
    electron_heating = ElectronHeatingConstantRFPower(species, POWER_RF_W, chamber)
    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name="bo_ei_point",
        log_folder_path=Path(__file__).resolve().parents[4].joinpath("outputs", "e09_bo_logs"),
        fast=FAST_MODE,
    )
    sol = model.solve(0.0, 1e-2, initial_state)
    return float(model.total_thrust(sol.y[:, -1]))


def _append_sample_to_dataset(x_row: list[float], y_thrust: float, meta: dict):
    data = read_json(GP_DATASET_PATH)
    samples = data.get("samples", [])
    samples.append({"x": x_row, "y_total_thrust_N": float(y_thrust), "meta": meta})
    data["samples"] = samples
    write_json(GP_DATASET_PATH, data)


def main() -> None:
    from train_gp_thrust import main as train_gp

    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)
    f107, f107a, ap = _space_weather_params(DATE, records, f107a_map)
    orbit_pts = _orbit_points(ALTITUDE_KM, INCLINATION_DEG, RAAN_DEG, ORBIT_POINTS, DATE)

    train_gp()
    gp, x_mean, x_std = _load_gp()

    log_rows = []
    best_mu = -np.inf
    for it in range(1, N_ITER + 1):
        candidates = []
        for area in A_CANDIDATES_M2:
            g_mu, g_sigma, idx_star = _evaluate_orbit_score(area, orbit_pts, gp, x_mean, x_std, f107, f107a, ap)
            ei = _ei(g_mu, g_sigma, best_mu if np.isfinite(best_mu) else g_mu, XI)
            candidates.append(
                {
                    "area_m2": float(area),
                    "g_mu": g_mu,
                    "g_sigma": g_sigma,
                    "ei": ei,
                    "idx_star": idx_star,
                }
            )
        best_cand = max(candidates, key=lambda r: r["ei"])
        area_star = float(best_cand["area_m2"])
        t_idx = int(best_cand["idx_star"])
        p_star = orbit_pts[t_idx]

        true_thrust = _evaluate_true_thrust(p_star, area_star, f107, f107a, ap)
        n2, o2, o, n, t_k, _ = _msis_with_rho(ALTITUDE_KM, p_star["lat"], p_star["lon"], p_star["date"], f107, f107a, ap)
        x_new = make_feature_row(
            n2,
            o2,
            o,
            n,
            t_k,
            ARGON_INJECTION_RATE,
            area_star,
            rho_kg_m3=rho_from_species_densities(n2, o2, o, n),
            speed_m_s=_orbital_speed(ALTITUDE_KM),
        )
        _append_sample_to_dataset(
            x_new,
            true_thrust,
            {
                "orbit_id": "default_orbit",
                "altitude_km": ALTITUDE_KM,
                "inclination_deg": INCLINATION_DEG,
                "raan_deg": RAAN_DEG,
                "theta_rad": p_star["theta"],
                "lat_deg": p_star["lat"],
                "lon_deg": p_star["lon"],
                "date": p_star["date"].isoformat(),
                "intake_area_m2": area_star,
                "eta_collection": ETA_COLLECTION,
                "success": True,
                "source": "bo_maxmin_ei",
            },
        )

        train_gp()
        gp, x_mean, x_std = _load_gp()
        best_mu = max(best_mu, float(best_cand["g_mu"]))

        row = {
            "iteration": it,
            "area_star_m2": area_star,
            "theta_star_rad": float(p_star["theta"]),
            "lat_star_deg": float(p_star["lat"]),
            "lon_star_deg": float(p_star["lon"]),
            "predicted_g_mu": float(best_cand["g_mu"]),
            "predicted_g_sigma": float(best_cand["g_sigma"]),
            "predicted_ei": float(best_cand["ei"]),
            "true_thrust_N": float(true_thrust),
        }
        log_rows.append(row)
        print(f"[{it}/{N_ITER}] {row}")

    write_json(
        BO_LOG_PATH,
        {
            "settings": {
                "altitude_km": ALTITUDE_KM,
                "inclination_deg": INCLINATION_DEG,
                "raan_deg": RAAN_DEG,
                "orbit_points": ORBIT_POINTS,
                "argon_injection_rate": ARGON_INJECTION_RATE,
                "eta_collection": ETA_COLLECTION,
                "n_iter": N_ITER,
                "xi": XI,
            },
            "iterations": log_rows,
        },
    )
    print(f"Saved BO log: {BO_LOG_PATH}")


if __name__ == "__main__":
    main()
