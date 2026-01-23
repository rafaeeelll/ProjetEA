import sys
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi


# If global_model_package is not installed as package with pip install -e . , adds the global_model_package to the path so that it can be imported as a package
try:
    import global_model_package  # noqa: F401
    print("'global_model_package' imported as pip package or already in sys.path.")
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.reactions import ElectronHeatingConstantRFPower

from reaction_set_N_et_O import get_species_and_reactions, get_neutral_atmosphere


# --- Sweep settings ---
altitudes_km = np.arange(150, 301, 10)
power_w = 3000

# --- MSIS defaults (arbitrary) ---
date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0

# --- Target pressure from average at 250 km ---
atm_ref = get_neutral_atmosphere(250, lat=lat, lon=lon, date=date)
target_pressure = max(atm_ref["pressure_pa"], 1e-3)
print(f"Using target_pressure={target_pressure:.3e} Pa (source={atm_ref.get('source')})")

# --- Chamber config (target pressure mode) ---
config_dict = {
    "R": 6e-2,
    "L": 10e-2,
    "target_pressure": target_pressure,
    "omega": 13.56e6 * 2 * pi,
    "N": 5,
    "R_coil": 2,
}

log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("outputs", "logs_for_thrust_by_altitude")
os.makedirs(log_folder_path, exist_ok=True)

results = {
    "altitudes_km": [],
    "ion_thrust_N": [],
    "neutral_thrust_N": [],
    "total_thrust_N": [],
    "power_w": power_w,
    "target_pressure_pa": target_pressure,
    "msis_date": date.isoformat(),
    "msis_lat": lat,
    "msis_lon": lon,
}

for altitude in altitudes_km:
    atm = get_neutral_atmosphere(altitude, lat=lat, lon=lon, date=date)

    chamber = Chamber(config_dict)
    # Provide grid parameters for thrust computation only
    chamber.V_grid = 1000
    chamber.beta_i = 0.7
    chamber.beta_g = 0.3

    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber,
        altitude,
        lat=lat,
        lon=lon,
        date=date,
        ion_seed=1e10,
        electron_seed=1e12,
        atm=atm,
    )

    electron_heating = ElectronHeatingConstantRFPower(species, power_w, chamber)
    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name=f"NO_Ar_thrust_alt_{altitude}km",
        log_folder_path=log_folder_path,
    )

    try:
        print(f"Solving model for altitude={altitude} km...")
        sol = model.solve(0, 1, initial_state)
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

    results["altitudes_km"].append(float(altitude))
    results["ion_thrust_N"].append(float(ion_thrust))
    results["neutral_thrust_N"].append(float(neutral_thrust))
    results["total_thrust_N"].append(float(total_thrust))

with open(log_folder_path.joinpath("thrust_vs_altitude.json"), "w") as file:
    json.dump(results, file, indent=2)

plt.plot(results["altitudes_km"], results["ion_thrust_N"], marker="o", label="Ion thrust")
plt.plot(results["altitudes_km"], results["total_thrust_N"], marker="s", label="Total thrust")
plt.xlabel("Altitude (km)")
plt.ylabel("Thrust (N)")
plt.title("Thrust vs Altitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(log_folder_path.joinpath("thrust_vs_altitude.pdf"), bbox_inches="tight")
