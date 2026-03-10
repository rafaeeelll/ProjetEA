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

try:
    import global_model_package 
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parents[3].joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.chamber_caracteristics import Chamber
from global_model_package.model import GlobalModel
from global_model_package.reactions import ElectronHeatingConstantRFPower

from msis_densities import (
    _compute_f107a,
    _parse_space_weather,
    _space_weather_params,
    SPACE_WEATHER_PATH,
    get_msis_neutral_atmosphere,
)
from reaction_set_N_et_O import get_species_and_reactions
from Drag.drag_model import (
    A_BODY_M2_DEFAULT,
    CD_BODY_DEFAULT,
    collection_efficiency,
    drag_total_fmf,
    mass_density_from_number_densities,
)

# --- Fixed point on orbit ---
theta_fixed_rad = 0.0

# --- Orbit / MSIS settings ---
date = datetime(2020, 1, 1, 12, 0, 0)
inclination_deg = 51.6
raan_deg = 0.0

# Sweep altitude
ALTITUDE_SWEEP_POINTS = int(os.environ.get("ALTITUDE_SWEEP_POINTS", "101"))
altitudes_km = np.linspace(150.0, 250.0, ALTITUDE_SWEEP_POINTS)

ARGON_INJECTION_RATE = 0.0

# Global model settings
POWER_RF_W = 1000.0
ION_SEED = 1e10
ELECTRON_SEED = 1e12
A_INTAKE_M2 = 0.23
FAST_MODE = True
T0 = 0.0
TF = 1e-2

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

# --- Space weather ---
records = _parse_space_weather(SPACE_WEATHER_PATH)
f107a_map = _compute_f107a(records)

# Drag constants
CD_BODY = CD_BODY_DEFAULT
A_BODY_M2 = A_BODY_M2_DEFAULT

# Earth
_mu_earth = 3.986004418e14  # m^3/s^2
_r_earth = 6371e3  # m
omega_earth = 7.2921159e-5  # rad/s
inc = np.radians(inclination_deg)
raan = np.radians(raan_deg)

# Storage
drag_N = []
thrust_N = []
thrust_minus_drag_N = []
thrust_time_series_last_N = []
final_state_summary = []

sim_log_dir = Path(__file__).resolve().parents[4].joinpath("figures", "E09", "drag", "simulation_logs")
os.makedirs(sim_log_dir, exist_ok=True)

for altitude_km in altitudes_km:
    radius = _r_earth + altitude_km * 1000.0
    orbital_speed = np.sqrt(_mu_earth / radius)
    mean_motion = np.sqrt(_mu_earth / radius ** 3)

    t = theta_fixed_rad / mean_motion
    dt = date + timedelta(seconds=float(t))

    x_orb = radius * np.cos(theta_fixed_rad)
    y_orb = radius * np.sin(theta_fixed_rad)

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

    f107, f107a, ap = _space_weather_params(dt, records, f107a_map)
    atm = get_msis_neutral_atmosphere(
        altitude_km,
        lat=lat_deg,
        lon=lon_deg,
        date=dt,
        f107=f107,
        f107a=f107a,
        ap=ap,
    )

    n_N2 = float(atm["N2"])
    n_O2 = float(atm["O2"])
    n_O = float(atm["O"])
    n_N = float(atm["N"])
    eta_collection_eff = collection_efficiency(A_intake=A_INTAKE_M2)
    rho = mass_density_from_number_densities(
        n2_m3=n_N2,
        o2_m3=n_O2,
        o_m3=n_O,
        n_m3=n_N,
    )
    drag = drag_total_fmf(
        rho=rho,
        speed_m_s=orbital_speed,
        intake_area_m2=A_INTAKE_M2,
        cd_body=CD_BODY,
        a_body_m2=A_BODY_M2,
    )

    chamber = Chamber(CHAMBER_CONFIG)
    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber,
        float(altitude_km),
        argon_injection_rate=ARGON_INJECTION_RATE,
        lat=lat_deg,
        lon=lon_deg,
        date=dt,
        ion_seed=ION_SEED,
        electron_seed=ELECTRON_SEED,
        A_intake=A_INTAKE_M2,
        eta_collection=eta_collection_eff,
        atm=atm,
    )
    electron_heating = ElectronHeatingConstantRFPower(species, POWER_RF_W, chamber)
    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name=f"drag_altitude_{float(altitude_km):.1f}",
        log_folder_path=sim_log_dir,
        fast=FAST_MODE,
    )

    sol = model.solve(T0, TF, initial_state)
    states = np.asarray(sol.y, dtype=float).T

    total_thrust_series = [float(model.total_thrust(state)) for state in states]
    thrust_from_series_last = float(total_thrust_series[-1])
    final_state = np.asarray(sol.y[:, -1], dtype=float)
    thrust_from_final_state = float(model.total_thrust(final_state))

    drag_N.append(drag)
    thrust_N.append(thrust_from_final_state)
    thrust_minus_drag_N.append(thrust_from_final_state - drag)
    thrust_time_series_last_N.append(thrust_from_series_last)

    final_state_summary.append(
        {
            "altitude_km": float(altitude_km),
            "solver_success": bool(sol.success),
            "solver_status": int(sol.status),
            "nfev": int(sol.nfev),
            "final_total_thrust_N": thrust_from_final_state,
            "time_series_last_total_thrust_N": thrust_from_series_last,
            "T_e_eV": float(final_state[species.nb]),
            "T_mono_eV": float(final_state[species.nb + 1]),
            "T_diato_eV": float(final_state[species.nb + 2]),
            "eta_collection_eff": float(eta_collection_eff),
        }
    )


# --- Output folder dedicated to drag ----
out_dir = Path(__file__).resolve().parents[4].joinpath("figures", "E09", "drag")
os.makedirs(out_dir, exist_ok=True)

# Save raw data
results = {
    "altitudes_km": altitudes_km.tolist(),
    "drag_N": drag_N,
    "thrust_N": thrust_N,
    "thrust_minus_drag_N": thrust_minus_drag_N,
    "thrust_time_series_last_N": thrust_time_series_last_N,
    "final_state_summary": final_state_summary,
    "theta_fixed_rad": theta_fixed_rad,
    "argon_rate": ARGON_INJECTION_RATE,
    "power_rf_W": POWER_RF_W,
    "A_intake_m2": A_INTAKE_M2,
    "cd_body": CD_BODY,
    "body_area_m2": A_BODY_M2,
    "eta_collection_eff": float(collection_efficiency(A_intake=A_INTAKE_M2)),
    "source": "GlobalModel solve -> time_series -> final_state",
}
with out_dir.joinpath("drag_thrust_vs_altitude.json").open("w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2)

# Combined figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(altitudes_km, drag_N, color="tab:red")
axes[0].set_title("Drag vs Altitude")
axes[0].set_xlabel("Altitude (km)")
axes[0].set_ylabel("Drag (N)")
axes[0].grid(True)

axes[1].plot(altitudes_km, thrust_N, color="tab:blue")
axes[1].set_title("Thrust vs Altitude")
axes[1].set_xlabel("Altitude (km)")
axes[1].set_ylabel("Thrust (N)")
axes[1].grid(True)

axes[2].plot(altitudes_km, thrust_minus_drag_N, color="tab:green")
axes[2].set_title("Thrust - Drag vs Altitude")
axes[2].set_xlabel("Altitude (km)")
axes[2].set_ylabel("Thrust - Drag (N)")
axes[2].grid(True)

fig.tight_layout()
fig.savefig(out_dir.joinpath("Fig_extra_drag_thrust_difference_vs_altitude.png"), dpi=200)

# Individual figures
fig_drag, ax_drag = plt.subplots(figsize=(7, 5))
ax_drag.plot(altitudes_km, drag_N, color="tab:red")
ax_drag.set_title("Drag vs Altitude")
ax_drag.set_xlabel("Altitude (km)")
ax_drag.set_ylabel("Drag (N)")
ax_drag.grid(True)
fig_drag.tight_layout()
fig_drag.savefig(out_dir.joinpath("Fig_2_10_drag_vs_altitude.png"), dpi=200)

fig_thrust, ax_thrust = plt.subplots(figsize=(7, 5))
ax_thrust.plot(altitudes_km, thrust_N, color="tab:blue")
ax_thrust.set_title("Thrust vs Altitude")
ax_thrust.set_xlabel("Altitude (km)")
ax_thrust.set_ylabel("Thrust (N)")
ax_thrust.grid(True)
fig_thrust.tight_layout()
fig_thrust.savefig(out_dir.joinpath("Fig_3_5_thrust_vs_altitude.png"), dpi=200)

fig_delta, ax_delta = plt.subplots(figsize=(7, 5))
ax_delta.plot(altitudes_km, thrust_minus_drag_N, color="tab:green")
ax_delta.set_title("Thrust - Drag vs Altitude")
ax_delta.set_xlabel("Altitude (km)")
ax_delta.set_ylabel("Thrust - Drag (N)")
ax_delta.grid(True)
fig_delta.tight_layout()
fig_delta.savefig(out_dir.joinpath("Fig_3_10c_thrust_minus_drag_vs_altitude.png"), dpi=200)

plt.show()
