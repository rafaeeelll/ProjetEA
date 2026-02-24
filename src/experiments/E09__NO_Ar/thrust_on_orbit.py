import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi


try:
    import global_model_package
    print("'global_model_package' imported as pip package or already in sys.path.")
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.reactions import ElectronHeatingConstantRFPower

from reaction_set_N_et_O import get_species_and_reactions, get_neutral_atmosphere
from msis_densities import _compute_f107a, _parse_space_weather, _space_weather_params, SPACE_WEATHER_PATH


# --- Orbital sweep settings ---
altitude_km = 183.0
power_w = 1000
FAST_MODE = True

# --- Orbit / MSIS inputs ---
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

# --- Reference atmosphere at 250 km ---
atm_ref = get_neutral_atmosphere(altitude_km, lat=lat, lon=lon, date=date)
print(f"Reference MSIS pressure at 183 km: {atm_ref['pressure_pa']:.3e} Pa")

# --- Chamber config (gridded thruster mode) ---
config_dict = {
    "R": 6e-2,
    "L": 10e-2,
    "V_grid": 1000,
    "beta_i": 0.7,
    "beta_g": 0.3,
    "omega": 13.56e6 * 2 * pi,
    "N": 5,
    "R_coil": 2,
}

log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("outputs", "logs_for_thrust_by_altitude")
os.makedirs(log_folder_path, exist_ok=True)

results = {
    "angles_rad": [],
    "with_argon": {
        "ion_thrust_N": [],
        "neutral_thrust_N": [],
        "total_thrust_N": [],
    },
    "without_argon": {
        "ion_thrust_N": [],
        "neutral_thrust_N": [],
        "total_thrust_N": [],
    },
    "power_w": power_w,
    "msis_date": date.isoformat(),
    "msis_lat": lat,
    "msis_lon": lon,
    "altitude_km": altitude_km,
    "orbit_points": orbit_points,
    "inclination_deg": inclination_deg,
    "raan_deg": raan_deg,
    "V_grid": config_dict["V_grid"],
    "beta_i": config_dict["beta_i"],
    "beta_g": config_dict["beta_g"],
}


def run_case(argon_injection_rate: float, case_key: str, case_label: str) -> None:
    print(f"\n=== Running case: {case_label} (argon_injection_rate={argon_injection_rate:.2e}) ===")
    mu_earth = 3.986004418e14  # m^3/s^2
    r_earth = 6371e3  # m
    omega_earth = 7.2921159e-5  # rad/s

    radius = r_earth + altitude_km * 1000.0
    mean_motion = np.sqrt(mu_earth / radius**3)

    angles = np.linspace(0.0, 2.0 * np.pi, orbit_points)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)

    if abs(np.sin(inc)) > 1e-8:
        sin_u0 = np.sin(np.radians(lat)) / np.sin(inc)
        sin_u0 = float(np.clip(sin_u0, -1.0, 1.0))
        u0 = float(np.arcsin(sin_u0))
    else:
        u0 = 0.0

    for theta in angles:
        u = u0 + theta
        t = u / mean_motion
        dt = date + timedelta(seconds=float(t))
        _f107, _f107a, _ap = _space_weather_params(dt, records, f107a_map)

        x_orb = radius * np.cos(u)
        y_orb = radius * np.sin(u)

        x_inc = x_orb
        y_inc = y_orb * np.cos(inc)
        z_inc = y_orb * np.sin(inc)

        x_eci = x_inc * np.cos(raan) - y_inc * np.sin(raan)
        y_eci = x_inc * np.sin(raan) + y_inc * np.cos(raan)
        z_eci = z_inc

        cos_earth = np.cos(omega_earth * t)
        sin_earth = np.sin(omega_earth * t)
        x_ecef = x_eci * cos_earth + y_eci * sin_earth
        y_ecef = -x_eci * sin_earth + y_eci * cos_earth
        z_ecef = z_eci

        lat_rad = np.arcsin(z_ecef / radius)
        lon_rad = np.arctan2(y_ecef, x_ecef)

        lat_deg = float(np.degrees(lat_rad))
        lon_deg = float(np.degrees(lon_rad))

        atm = get_neutral_atmosphere(
            altitude_km,
            lat=lat_deg,
            lon=lon_deg,
            date=dt,
            f107=_f107,
            f107a=_f107a,
            ap=_ap,
        )
        chamber = Chamber(config_dict)

        species, initial_state, reactions_list, _ = get_species_and_reactions(
            chamber,
            altitude_km,
            lat=lat_deg,
            lon=lon_deg,
            date=dt,
            ion_seed=1e10,
            electron_seed=1e12,
            argon_injection_rate=argon_injection_rate,
            atm=atm,
        )

        electron_heating = ElectronHeatingConstantRFPower(species, power_w, chamber)
        model = GlobalModel(
            species,
            reactions_list,
            chamber,
            electron_heating,
            simulation_name=f"NO_Ar_{case_key}_angle_{theta:.3f}",
            log_folder_path=log_folder_path,
            fast=FAST_MODE,
        )

        try:
            print(f"Solving model for theta={theta:.3f} rad (lat={lat_deg:.2f}, lon={lon_deg:.2f})...")
            sol = model.solve(0, 1e-2, initial_state)
            print("Model resolved!")
        except Exception as exception:
            print("Entering exception...")
            model.var_tracker.save_tracked_variables()
            print("Variables saved")
            raise exception

        final_state = sol.y[:, -1]
        ion_thrust = model.total_ion_thrust(final_state)
        neutral_thrust = model.total_neutral_thrust(final_state)
        total_thrust = model.total_thrust(final_state)

        results["angles_rad"].append(float(theta))
        results[case_key]["ion_thrust_N"].append(float(ion_thrust))
        results[case_key]["neutral_thrust_N"].append(float(neutral_thrust))
        results[case_key]["total_thrust_N"].append(float(total_thrust))


run_case(argon_injection_rate=1e18, case_key="with_argon", case_label="With Argon")
run_case(argon_injection_rate=0.0, case_key="without_argon", case_label="Without Argon")

with open(log_folder_path.joinpath("thrust_vs_orbit.json"), "w") as file:
    json.dump(results, file, indent=2)

plt.figure(figsize=(8, 5))
angles = np.asarray(results["angles_rad"], dtype=float)
with_argon = results.get("with_argon", {})
without_argon = results.get("without_argon", {})

with_ion = np.asarray(with_argon.get("ion_thrust_N", []), dtype=float)
with_total = np.asarray(with_argon.get("total_thrust_N", []), dtype=float)
wo_ion = np.asarray(without_argon.get("ion_thrust_N", []), dtype=float)
wo_total = np.asarray(without_argon.get("total_thrust_N", []), dtype=float)

if angles.size == 0:
    n = max(with_ion.size, with_total.size, wo_ion.size, wo_total.size)
    angles = np.linspace(0.0, 2.0 * np.pi, n) if n > 0 else angles

if with_ion.size:
    plt.plot(angles[: with_ion.size], with_ion, marker="o", label="Ion thrust (Ar)")
if with_total.size:
    plt.plot(angles[: with_total.size], with_total, marker="s", label="Total thrust (Ar)")
if wo_ion.size:
    plt.plot(
        angles[: wo_ion.size],
        wo_ion,
        marker="o",
        linestyle="--",
        label="Ion thrust (no Ar)",
    )
if wo_total.size:
    plt.plot(
        angles[: wo_total.size],
        wo_total,
        marker="s",
        linestyle="--",
        label="Total thrust (no Ar)",
    )
plt.xlabel("Angle orbital (rad)")
plt.ylabel("Thrust (N)")
plt.title("Thrust vs Angle (With vs Without Argon)")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(log_folder_path.joinpath("thrust_vs_orbit.png"), dpi=300)
plt.show()
