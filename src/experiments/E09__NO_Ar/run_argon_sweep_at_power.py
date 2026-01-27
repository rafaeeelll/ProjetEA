import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi


try:
    import global_model_package  # noqa: F401
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.reactions import ElectronHeatingConstantRFPower

from reaction_set_N_et_O import get_species_and_reactions, get_neutral_atmosphere


# --- Fixed operating point ---
altitude_km = 250
power_w = 1500  # set this to the CL-limited power you found
t0, tf = 0.0, 1e-2

date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0

# Argon injection sweep (particles/s)
argon_injection_rates = np.logspace(16, 19, 7)

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

log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath(
    "outputs", "logs_argon_sweep_at_power"
)
log_folder_path.mkdir(parents=True, exist_ok=True)

atm = get_neutral_atmosphere(altitude_km, lat=lat, lon=lon, date=date)

total_thrust_list = []
ion_thrust_list = []

for argon_rate in argon_injection_rates:
    chamber = Chamber(config_dict)
    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber,
        altitude_km,
        lat=lat,
        lon=lon,
        date=date,
        argon_injection_rate=float(argon_rate),
        atm=atm,
    )

    electron_heating = ElectronHeatingConstantRFPower(species, power_w, chamber)
    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name=f"E09_Ar_sweep_{argon_rate:.2e}",
        log_folder_path=log_folder_path,
    )

    sol = model.solve(t0, tf, initial_state)
    final_state = sol.y[:, -1]

    ion_thrust = model.total_ion_thrust(final_state)
    total_thrust = model.total_thrust(final_state)

    ion_thrust_list.append(float(ion_thrust))
    total_thrust_list.append(float(total_thrust))

plt.figure(figsize=(8, 5))
plt.semilogx(argon_injection_rates, ion_thrust_list, marker="o", label="Ion thrust")
plt.semilogx(argon_injection_rates, total_thrust_list, marker="s", label="Total thrust")
plt.xlabel("Argon injection rate (s$^{-1}$)")
plt.ylabel("Thrust (N)")
plt.title(f"Thrust vs Argon Injection @ {power_w} W, {altitude_km} km")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
