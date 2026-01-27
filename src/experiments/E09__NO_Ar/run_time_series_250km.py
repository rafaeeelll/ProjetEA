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


altitude_km = 250
power_w = 3000
t0, tf = 0.0, 1.0

date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0

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

atm = get_neutral_atmosphere(altitude_km, lat=lat, lon=lon, date=date)
chamber = Chamber(config_dict)
species, initial_state, reactions_list, _ = get_species_and_reactions(
    chamber,
    altitude_km,
    lat=lat,
    lon=lon,
    date=date,
    ion_seed=1e15,
    electron_seed=1e20,
    atm=atm,
)

electron_heating = ElectronHeatingConstantRFPower(species, power_w, chamber)
log_folder_path = Path(__file__).resolve().parent.parent.parent.parent.joinpath("outputs", "logs_time_series_250km")
log_folder_path.mkdir(parents=True, exist_ok=True)

model = GlobalModel(
    species,
    reactions_list,
    chamber,
    electron_heating,
    simulation_name="E09_time_series_250km",
    log_folder_path=log_folder_path,
)

sol = model.solve(t0, tf, initial_state)
time_points = sol.t
states = sol.y

# Same structure as E02: species on top, temperatures on bottom
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

for i, sp in enumerate(species.species):
    ax1.semilogy(time_points, states[i], label=sp.name)
ax1.set_ylabel("Density (m$^{-3}$)")
ax1.set_title(f"E09 Species Concentrations Over Time at {altitude_km} km")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(ncol=3, fontsize=8, loc="best")

ax3 = ax2.twinx()
ax2.plot(time_points, states[species.nb], label="Electron Temp (eV)", color="blue")
for i in range(1, 3):
    ax3.plot(
        time_points,
        states[species.nb + i],
        linestyle="--",
        label=f"Molecules with {i} atoms Temp (eV)",
    )

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Electron Temperature (eV)", color="blue")
ax3.set_ylabel("Molecules Temperature (eV)", color="red")
ax2.tick_params(axis="y", labelcolor="blue")
ax3.tick_params(axis="y", labelcolor="red")
ax2.set_title("Temperature Evolution")
ax2.grid(True, alpha=0.3)

lines_2, labels_2 = ax2.get_legend_handles_labels()
lines_3, labels_3 = ax3.get_legend_handles_labels()
ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc="best", fontsize=8)

plt.tight_layout()
plt.savefig(log_folder_path.joinpath("species_and_temperature_vs_time_250km.pdf"), bbox_inches="tight")
