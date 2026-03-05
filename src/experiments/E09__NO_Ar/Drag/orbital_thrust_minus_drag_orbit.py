import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

E09_DIR = Path(__file__).resolve().parents[1]
if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

# MSIS helpers from existing code
from msis_densities import (
    _compute_f107a,
    _parse_space_weather,
    _space_weather_params,
    SPACE_WEATHER_PATH,
)
from Drag.drag_model import (
    A_BODY_M2_DEFAULT,
    CD_BODY_DEFAULT,
    drag_total_fmf,
    mass_density_from_number_densities,
)

try:
    from nrlmsise00 import msise_model  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise RuntimeError(
        f"nrlmsise00 is required for drag/thrust sweep. Install with `pip install nrlmsise00`: {exc}"
    ) from exc

# model loader (same as in orbital_thrust_minus_drag)
def _load_model() -> tuple:
    model_path = (
        Path(__file__)
        .resolve()
        .parents[4]
        .joinpath("outputs", "thrust_dataset", "thrust_rbf_model.npz")
    )
    meta_path = model_path.with_suffix(".json")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    data = np.load(model_path)
    from scipy.interpolate import RBFInterpolator
    rbf = RBFInterpolator(
        data["Xs"],
        data["y"],
        kernel=str(data["kernel"]),
        smoothing=float(data["smoothing"]),
        neighbors=int(data["neighbors"]),
    )
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    return rbf, data["x_mean"], data["x_std"], meta


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _predict_rbf_row(rbf, x_row: np.ndarray) -> float:
    x_row = x_row.reshape(1, -1)
    try:
        return float(rbf(x_row)[0])
    except Exception:
        jitter = 1e-12 * np.random.randn(*x_row.shape)
        try:
            return float(rbf(x_row + jitter)[0])
        except Exception:
            return 0.0

# --- Orbit / MSIS settings ---
date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0
inclination_deg = 51.6
raan_deg = lon
orbit_points = 90

# altitudes to sweep (km) – only four for readability
altitudes_km = np.array([180.0, 190.0, 200.0, 210.0])

# argon injection rates to evaluate
argon_rates = [0.0, 1e17, 1e18, 5e18]
argon_cases = [(rate, f"{rate:.1e}") for rate in argon_rates]

# --- Space weather ---
records = _parse_space_weather(SPACE_WEATHER_PATH)
f107a_map = _compute_f107a(records)
f107, f107a, ap = _space_weather_params(date, records, f107a_map)

# drag constants
INTAKE_AREA_M2 = 0.5
CD_BODY = CD_BODY_DEFAULT
A_BODY_M2 = A_BODY_M2_DEFAULT

# Earth
_mu_earth = 3.986004418e14  # m^3/s^2
_r_earth = 6371e3  # m

# compute orbit geometry constants that do not depend on altitude
theta_array = np.linspace(0.0, 2.0 * pi, orbit_points, endpoint=False)
omega_earth = 7.2921159e-5  # rad/s
inc = np.radians(inclination_deg)
raan = np.radians(raan_deg)

# load surrogate model
rbf, x_mean, x_std, _ = _load_model()

# set up result storage for all altitudes
results_all = {}

# prepare subplots in a 2x2 grid for readability
n_alt = len(altitudes_km)
cols = 2
rows = int(np.ceil(n_alt / cols))
fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), sharex=True, sharey=True)
# flatten axes for easy iteration
axes = axes.flatten()[:n_alt]

# loop over altitudes
for ax, altitude_km in zip(axes, altitudes_km):
    # containers for this altitude
    angles_rad = []
    thrust_N = {key: [] for _, key in argon_cases}
    drag_N = []
    thrust_minus_drag_N = {key: [] for _, key in argon_cases}

    # altitude-specific geometry
    radius = _r_earth + altitude_km * 1000.0
    orbital_speed = np.sqrt(_mu_earth / radius)
    mean_motion = np.sqrt(_mu_earth / radius ** 3)

    for theta in theta_array:
        t = theta / mean_motion
        dt = date + timedelta(seconds=float(t))

        # compute position as before
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

        lat_rad = np.arcsin(z_ecef / radius)
        lon_rad = np.arctan2(y_ecef, x_ecef)
        lat_deg = float(np.degrees(lat_rad))
        lon_deg = float(np.degrees(lon_rad))

        # MSIS density for drag
        dens, temp = msise_model(dt, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
        dens = np.array(dens, dtype=float)
        n_N2 = dens[2] * 1e6
        n_O2 = dens[3] * 1e6
        n_O = dens[1] * 1e6
        n_N = dens[7] * 1e6
        rho = mass_density_from_number_densities(
            n2_m3=n_N2,
            o2_m3=n_O2,
            o_m3=n_O,
            n_m3=n_N,
        )
        drag = drag_total_fmf(
            rho=rho,
            speed_m_s=orbital_speed,
            intake_area_m2=INTAKE_AREA_M2,
            cd_body=CD_BODY,
            a_body_m2=A_BODY_M2,
        )

        # record angle and drag once per step
        angles_rad.append(float(theta))
        drag_N.append(drag)

        # base features
        n2 = n_N2
        o2 = n_O2
        o = n_O
        n = n_N
        T_K = float(temp[1])

        for rate, key in argon_cases:
            feat = np.array([
                _safe_log10(np.array([n2]))[0],
                _safe_log10(np.array([o2]))[0],
                _safe_log10(np.array([o]))[0],
                _safe_log10(np.array([n]))[0],
                T_K,
                _safe_log10(np.array([rate]))[0],
            ], dtype=float)
            Xs = (feat - x_mean) / x_std
            thrust_val = _predict_rbf_row(rbf, Xs)
            thrust_N[key].append(thrust_val)
            thrust_minus_drag_N[key].append(thrust_val - drag)

    # store results for this altitude
    results_all[f"{altitude_km:.1f}"] = {
        "angles_rad": angles_rad,
        "drag_N": drag_N,
        "thrust_N": thrust_N,
        "thrust_minus_drag_N": thrust_minus_drag_N,
        "altitude_km": altitude_km,
    }

    # plot on axis
    for rate, key in argon_cases:
        ax.plot(angles_rad, thrust_minus_drag_N[key], marker="o", label=f"rate={rate:.1e}")
    ax.set_title(f"altitude {altitude_km:.1f} km")
    ax.set_xlabel("Angle orbital (rad)")
    ax.set_ylabel("Thrust minus drag (N)")
    ax.grid(True)
    ax.legend()

# save combined results
out_path = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "logs_for_thrust_minus_drag_orbit")
)
os.makedirs(out_path, exist_ok=True)
with open(out_path.joinpath("thrust_minus_drag_orbit.json"), "w") as fp:
    json.dump(results_all, fp, indent=2)

plt.tight_layout()
plt.show()
