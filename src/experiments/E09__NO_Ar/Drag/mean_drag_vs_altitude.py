import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

# MSIS helpers
E09_DIR = Path(__file__).resolve().parents[1]
if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

from msis_densities import (
    _compute_f107a,
    _parse_space_weather,
    _space_weather_params,
    SPACE_WEATHER_PATH,
)

# --- Orbital/drag constants ---
date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0
inclination_deg = 51.6
raan_deg = lon
orbit_points = 90

# altitudes to evaluate (km)
altitudes_km = np.arange(180.0, 221.0, 10.0)

# drag constants
CD = 2.2
CROSS_SECTION_AREA_M2 = 1.0

# species masses
_m_N2 = 4.65e-26
_m_O2 = 5.31e-26
_m_O = 2.67e-26
_m_N = 2.33e-26

# Earth constants
_mu_earth = 3.986004418e14  # m^3/s^2
_r_earth = 6371e3  # m

# space weather
records = _parse_space_weather(SPACE_WEATHER_PATH)
f107a_map = _compute_f107a(records)
f107, f107a, ap = _space_weather_params(date, records, f107a_map)

# precompute orbit geometry constants
omega_earth = 7.2921159e-5  # rad/s
inc = np.radians(inclination_deg)
raan = np.radians(raan_deg)

# results
mean_drag_list = []

for alt in altitudes_km:
    radius = _r_earth + alt * 1000.0
    mean_motion = np.sqrt(_mu_earth / radius**3)
    orbital_speed = np.sqrt(_mu_earth / radius)

    drags = []
    for theta in np.linspace(0.0, 2.0 * pi, orbit_points, endpoint=False):
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

        lat_rad = np.arcsin(z_ecef / radius)
        lon_rad = np.arctan2(y_ecef, x_ecef)
        lat_deg = float(np.degrees(lat_rad))
        lon_deg = float(np.degrees(lon_rad))

        from nrlmsise00 import msise_model  # type: ignore
        dens, temp = msise_model(dt, alt, lat_deg, lon_deg, f107a, f107, ap)
        dens = np.array(dens, dtype=float)
        n_N2 = dens[2] * 1e6
        n_O2 = dens[3] * 1e6
        n_O = dens[1] * 1e6
        n_N = dens[7] * 1e6
        rho = n_N2 * _m_N2 + n_O2 * _m_O2 + n_O * _m_O + n_N * _m_N
        drag = 0.5 * rho * orbital_speed ** 2 * CD * CROSS_SECTION_AREA_M2
        drags.append(drag)

    mean_drag = float(np.mean(drags)) if drags else 0.0
    mean_drag_list.append(mean_drag)

# save
output = {
    "altitudes_km": altitudes_km.tolist(),
    "mean_drag_N": mean_drag_list,
    "orbit_points": orbit_points,
    "inclination_deg": inclination_deg,
    "raan_deg": raan_deg,
    "msis_date": date.isoformat(),
}

out_path = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "logs_for_drag_vs_altitude")
)
os.makedirs(out_path, exist_ok=True)
with open(out_path.joinpath("mean_drag_vs_altitude.json"), "w") as fp:
    json.dump(output, fp, indent=2)

# plot
plt.figure(figsize=(8, 5))
plt.plot(altitudes_km, mean_drag_list, marker="o")
plt.xlabel("Altitude (km)")
plt.ylabel("Mean drag (N)")
plt.title("Mean drag vs altitude")
plt.grid(True)
plt.tight_layout()
plt.show()
