import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

# the existing helpers used by orbital_thrust for space weather and MSIS
E09_DIR = Path(__file__).resolve().parents[1]
if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

from msis_densities import (
    _compute_f107a,
    _parse_space_weather,
    _space_weather_params,
    SPACE_WEATHER_PATH,
)

# --- Orbit / MSIS settings (copied from orbital_thrust) ---
altitude_km = 183.0
date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0
inclination_deg = 51.6
raan_deg = lon
orbit_points = 90

# --- Space weather from file (F10.7, F10.7A, Ap) ---
records = _parse_space_weather(SPACE_WEATHER_PATH)
f107a_map = _compute_f107a(records)
f107, f107a, ap = _space_weather_params(date, records, f107a_map)

# --- constants for drag calculation ---
# simple constant drag coefficient * area approach; values taken from optimize_orbit
CD = 2.2
CROSS_SECTION_AREA_M2 = 1.0

# mass of species in kg
_m_N2 = 4.65e-26
_m_O2 = 5.31e-26
_m_O = 2.67e-26
_m_N = 2.33e-26

# Earth   
_mu_earth = 3.986004418e14  # m^3/s^2
_r_earth = 6371e3  # m

# compute orbital radius and speed (assume circular)
radius = _r_earth + altitude_km * 1000.0
orbital_speed = np.sqrt(_mu_earth / radius)  # m/s


def _msis_mass_density(
    alt_km: float,
    lat_deg: float,
    lon_deg: float,
    dt: datetime,
    f107: float,
    f107a: float,
    ap: float,
) -> float:
    """Return mass density (kg/m^3) from nrlmsise00 at the requested point."""
    try:
        from nrlmsise00 import msise_model  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"nrlmsise00 is required for drag computation. Install with `pip install nrlmsise00`: {exc}"
        )

    dens, temp = msise_model(dt, alt_km, lat_deg, lon_deg, f107a, f107, ap)
    dens = np.array(dens, dtype=float)

    # convert to number density (m^-3) and multiply by mass
    n_N2 = dens[2] * 1e6
    n_O2 = dens[3] * 1e6
    n_O = dens[1] * 1e6
    n_N = dens[7] * 1e6

    rho = (
        n_N2 * _m_N2
        + n_O2 * _m_O2
        + n_O * _m_O
        + n_N * _m_N
    )
    return float(rho)


def _compute_orbital_lat_lon(t: float) -> tuple[float, float]:
    """Return (lat_deg, lon_deg) of spacecraft after elapsed time t (s)."""
    # based on code in orbital_thrust
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    u0 = 0.0

    u = u0 + t * 0  # placeholder, we will compute angle externally
    # this helper doesn't actually get used; see main loop below
    return 0.0, 0.0


# --- run sweep and gather drag values ---
angles_rad = []
drag_N = []

# precompute some Earth rotation parameters
omega_earth = 7.2921159e-5  # rad/s
mean_motion = np.sqrt(_mu_earth / radius ** 3)

inc = np.radians(inclination_deg)
raan = np.radians(raan_deg)

for theta in np.linspace(0.0, 2.0 * pi, orbit_points, endpoint=False):
    # time since epoch
    t = theta / mean_motion
    dt = date + timedelta(seconds=float(t))

    # compute inertial coordinates
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

    rho = _msis_mass_density(altitude_km, lat_deg, lon_deg, dt, f107, f107a, ap)
    drag = 0.5 * rho * orbital_speed ** 2 * CD * CROSS_SECTION_AREA_M2

    angles_rad.append(float(theta))
    drag_N.append(drag)

# save results for later inspection
output = {
    "angles_rad": angles_rad,
    "drag_N": drag_N,
    "altitude_km": altitude_km,
    "orbit_points": orbit_points,
    "inclination_deg": inclination_deg,
    "raan_deg": raan_deg,
    "msis_date": date.isoformat(),
}

out_path = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "logs_for_drag_by_orbit")
)
os.makedirs(out_path, exist_ok=True)
with open(out_path.joinpath("drag_vs_orbit.json"), "w") as fp:
    json.dump(output, fp, indent=2)

# simple plot
plt.figure(figsize=(8, 5))
angles = np.asarray(angles_rad)
scatter = plt.plot(angles, np.asarray(drag_N), marker="o")
plt.xlabel("Angle orbital (rad)")
plt.ylabel("Drag force (N)")
plt.title("Estimated drag vs orbital angle (MSIS density)")
plt.grid(True)
plt.tight_layout()
plt.show()
