"""Etudie la variabilite orbitale et environnementale du cas E09.

On compare ici l'effet de l'angle orbital, de l'activite solaire et de la
geographie sur la marge propulsive du systeme.
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    OUT_DIR,
    Case,
    EARTH_MU,
    EARTH_RADIUS_M,
    build_sample_from_case,
    drag_components,
    solve_plasma_sample,
    weather_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
GP_MODEL_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_gp_model.pkl")
ARGON_EPS = 1e14


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _feature(sample: dict, n_features: int = 6) -> np.ndarray:
    full = np.array(
        [
            float(_safe_log10(np.array([sample["N2_m3"]]))[0]),
            float(_safe_log10(np.array([sample["O2_m3"]]))[0]),
            float(_safe_log10(np.array([sample["O_m3"]]))[0]),
            float(_safe_log10(np.array([sample["N_m3"]]))[0]),
            float(_safe_log10(np.array([sample["intake_area_m2"]]))[0]),
            float(_safe_log10(np.array([sample.get("argon_injection_rate", 0.0) + ARGON_EPS]))[0]),
        ],
        dtype=float,
    )
    return full[:n_features]


def _load_gp():
    if not GP_MODEL_PATH.exists():
        return None
    with GP_MODEL_PATH.open("rb") as handle:
        p = pickle.load(handle)
    return p["gp"], np.asarray(p["x_mean"], dtype=float), np.asarray(p["x_std"], dtype=float)


def _predict_thrust(sample: dict, gp_payload, fast_mode: bool, use_gp: bool) -> float:
    if use_gp and gp_payload is not None:
        gp, x_mean, x_std = gp_payload
        n_features = int(x_mean.shape[0])
        x = _feature(sample, n_features=n_features)
        xs = (x - x_mean) / x_std
        return float(gp.predict(xs.reshape(1, -1))[0])
    return float(
        solve_plasma_sample(
            sample,
            power_rf_w=1000.0,
            fast_mode=fast_mode,
            simulation_name="figE_pred",
        )["thrust_final_N"]
    )


def figure_14_orbit_variations(out_dir: Path, alt_km: float, inc_deg: float, raan_deg: float, area_m2: float, argon_rate: float, fast_mode: bool, use_gp: bool) -> None:
    records, f107a_map = weather_records()
    gp_payload = _load_gp()
    thetas = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    rho, drag, thrust, margin = [], [], [], []
    for th in thetas:
        sample = build_sample_from_case(
            Case(
                altitude_km=alt_km,
                area_m2=area_m2,
                argon_rate=argon_rate,
                inclination_deg=inc_deg,
                raan_deg=raan_deg,
                theta_rad=float(th),
            ),
            records=records,
            f107a_map=f107a_map,
        )
        d = drag_components(sample)
        t = _predict_thrust(sample, gp_payload, fast_mode=fast_mode, use_gp=use_gp)
        rho.append(d["rho"])
        drag.append(d["drag_total_N"])
        thrust.append(t)
        margin.append(t - d["drag_total_N"])

    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(thetas, rho)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("rho [kg/m^3]")
    ax[1].plot(thetas, drag)
    ax[1].set_ylabel("Drag [N]")
    ax[2].plot(thetas, thrust)
    ax[2].set_ylabel("Thrust [N]")
    ax[3].plot(thetas, margin)
    ax[3].set_ylabel("Margin [N]")
    ax[3].set_xlabel("True anomaly theta [rad]")
    for a in ax:
        a.set_xlim(0.0, 2.0 * np.pi)
        a.set_xticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi], ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_3_8_3_11_4_5_E14_orbit_variations.png"), dpi=220)
    plt.close(fig)


def figure_15_solar_activity(out_dir: Path, alt_km: float, area_m2: float, argon_rate: float, fast_mode: bool, use_gp: bool) -> None:
    from msis_densities import get_msis_neutral_atmosphere

    gp_payload = _load_gp()
    f107_vals = np.linspace(70.0, 200.0, 18)
    density = []
    margin = []
    speed = np.sqrt(EARTH_MU / (EARTH_RADIUS_M + alt_km * 1e3))
    for f107 in f107_vals:
        atm = get_msis_neutral_atmosphere(altitude_km=alt_km, lat=0.0, lon=0.0, date=datetime(2020, 1, 1, 12, 0, 0), f107=float(f107), f107a=float(f107), ap=15.0)
        sample = {
            "N2_m3": float(atm["N2"]),
            "O2_m3": float(atm["O2"]),
            "O_m3": float(atm["O"]),
            "N_m3": float(atm["N"]),
            "T_K": float(atm["T_K"]),
            "intake_area_m2": float(area_m2),
            "argon_injection_rate": float(argon_rate),
            "altitude_km": float(alt_km),
            "orbital_speed_m_s": float(speed),
        }
        d = drag_components(sample)
        t = _predict_thrust(sample, gp_payload, fast_mode=fast_mode, use_gp=use_gp)
        density.append(d["rho"])
        margin.append(t - d["drag_total_N"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(f107_vals, density)
    ax[0].set_yscale("log")
    ax[0].set_title("Density at fixed altitude")
    ax[0].set_ylabel("rho [kg/m^3]")
    ax[1].plot(f107_vals, margin)
    ax[1].set_title("Margin at fixed altitude")
    ax[1].set_ylabel("Margin [N]")
    for a in ax:
        a.set_xlabel("F10.7")
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_2_4_E15_solar_activity_effect.png"), dpi=220)
    plt.close(fig)


def figure_16_lat_lon_margin_map(out_dir: Path, alt_km: float, area_m2: float, argon_rate: float, fast_mode: bool, use_gp: bool) -> None:
    from msis_densities import get_msis_neutral_atmosphere

    gp_payload = _load_gp()
    lats = np.linspace(-80.0, 80.0, 17)
    lons = np.linspace(-180.0, 180.0, 37)
    speed = np.sqrt(EARTH_MU / (EARTH_RADIUS_M + alt_km * 1e3))
    margin = np.full((len(lats), len(lons)), np.nan)

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            atm = get_msis_neutral_atmosphere(altitude_km=alt_km, lat=float(lat), lon=float(lon), date=datetime(2020, 1, 1, 12, 0, 0))
            sample = {
                "N2_m3": float(atm["N2"]),
                "O2_m3": float(atm["O2"]),
                "O_m3": float(atm["O"]),
                "N_m3": float(atm["N"]),
                "T_K": float(atm["T_K"]),
                "intake_area_m2": float(area_m2),
                "argon_injection_rate": float(argon_rate),
                "altitude_km": float(alt_km),
                "orbital_speed_m_s": float(speed),
            }
            d = drag_components(sample)
            t = _predict_thrust(sample, gp_payload, fast_mode=fast_mode, use_gp=use_gp)
            margin[i, j] = t - d["drag_total_N"]

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    fig, ax = plt.subplots(figsize=(11, 4))
    pcm = ax.pcolormesh(lon_grid, lat_grid, margin, shading="nearest", cmap="RdYlGn")
    fig.colorbar(pcm, ax=ax, label="Margin [N]")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("Latitude/Longitude map of margin")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_extra_E16_lat_lon_margin_map.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section E figures: orbital variability.")
    parser.add_argument("--altitude-km", type=float, default=180.0)
    parser.add_argument("--inclination-deg", type=float, default=20.0)
    parser.add_argument("--raan-deg", type=float, default=-13.0)
    parser.add_argument("--area-m2", type=float, default=0.1)
    parser.add_argument("--argon-rate", type=float, default=0.0)
    parser.add_argument("--fast", action="store_true", help="Use fast mode if GP is unavailable.")
    parser.add_argument("--use-gp", action="store_true", help="Use the GP thrust model instead of the true plasma solve.")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_14_orbit_variations(out_dir, float(args.altitude_km), float(args.inclination_deg), float(args.raan_deg), float(args.area_m2), float(args.argon_rate), fast_mode=bool(args.fast), use_gp=bool(args.use_gp))
    figure_15_solar_activity(out_dir, float(args.altitude_km), float(args.area_m2), float(args.argon_rate), fast_mode=bool(args.fast), use_gp=bool(args.use_gp))
    figure_16_lat_lon_margin_map(out_dir, float(args.altitude_km), float(args.area_m2), float(args.argon_rate), fast_mode=bool(args.fast), use_gp=bool(args.use_gp))
    print(f"Saved Section E figures to {out_dir}")


if __name__ == "__main__":
    main()
