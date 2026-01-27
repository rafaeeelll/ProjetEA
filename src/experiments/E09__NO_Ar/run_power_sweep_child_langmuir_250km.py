import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, e, pi


try:
    import global_model_package  # noqa: F401
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.model import GlobalModel
from global_model_package.chamber_caracteristics import Chamber
from global_model_package.reactions import ElectronHeatingConstantRFPower

from reaction_set_N_et_O import get_species_and_reactions, get_neutral_atmosphere


# --- Scenario settings ---
altitude_km = 250
argon_injection_rate = 1e18  # particles / s
power_list = np.arange(200, 1500, 20)
t0, tf = 0.0, 1e-2

date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0

# Grid / CL parameters (adjust gap if you have a better value)
grid_gap_m = 1e-3

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
    "outputs", "logs_power_sweep_child_langmuir"
)
log_folder_path.mkdir(parents=True, exist_ok=True)

atm = get_neutral_atmosphere(altitude_km, lat=lat, lon=lon, date=date)

ion_current_density_list = []
total_thrust_list = []
ion_thrust_list = []
te_list = []
tmono_list = []
tdiato_list = []

for power_w in power_list:
    chamber = Chamber(config_dict)
    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber,
        altitude_km,
        lat=lat,
        lon=lon,
        date=date,
        argon_injection_rate=argon_injection_rate,
        atm=atm,
    )

    electron_heating = ElectronHeatingConstantRFPower(species, power_w, chamber)
    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name=f"E09_CL_sweep_{power_w}W",
        log_folder_path=log_folder_path,
    )

    sol = model.solve(t0, tf, initial_state)
    final_state = sol.y[:, -1]

    ion_current_density = model.total_ion_current(final_state)
    total_thrust = model.total_thrust(final_state)
    ion_thrust = model.total_ion_thrust(final_state)

    te = final_state[species.nb]
    tmono = final_state[species.nb + 1]
    tdiato = final_state[species.nb + 2]

    ion_current_density_list.append(float(ion_current_density))
    total_thrust_list.append(float(total_thrust))
    ion_thrust_list.append(float(ion_thrust))
    te_list.append(float(te))
    tmono_list.append(float(tmono))
    tdiato_list.append(float(tdiato))

# Child-Langmuir current limit (planar diode, singly charged ions)
try:
    ion_sp = species.get_specie_by_name("Ar+")
    q_ion = abs(ion_sp.charge)
    m_ion = ion_sp.mass
except Exception:
    q_ion = e
    m_ion = 6.63e-26

j_cl = (4 / 9) * epsilon_0 * np.sqrt(2 * q_ion / m_ion) * (config_dict["V_grid"] ** 1.5) / (grid_gap_m**2)

# Find first power where I_ion >= I_CL
power_limit = None
for p, j in zip(power_list, ion_current_density_list):
    if j >= j_cl:
        power_limit = p
        break

print(f"Child-Langmuir limit current density (Ar+, d={grid_gap_m} m): {j_cl:.3e} A/m^2")
if power_limit is None:
    print("No power in sweep reached CL limit.")
else:
    print(f"Approx. CL-limited power: {power_limit} W")

plt.figure(figsize=(8, 5))
plt.plot(power_list, ion_current_density_list, marker="o", label="Ion current density (model)")
plt.axhline(j_cl, color="red", linestyle="--", label="Child-Langmuir limit")
plt.xlabel("RF Power (W)")
plt.ylabel("Ion Current Density (A/m²)")
plt.title("Ion Current Density vs RF Power at 250 km")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Temperature vs power (electron + gas)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(power_list, te_list, marker="o", label="T_e (eV)", color="blue")
ax1.set_xlabel("RF Power (W)")
ax1.set_ylabel("Electron Temperature (eV)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(power_list, tmono_list, marker="s", linestyle="--", label="T_mono (eV)", color="red")
ax2.plot(power_list, tdiato_list, marker="^", linestyle="--", label="T_diato (eV)", color="green")
ax2.set_ylabel("Gas Temperatures (eV)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
plt.title("Temperatures vs RF Power at 250 km")
plt.tight_layout()
plt.show()

# Thrust vs power (mN)
plt.figure(figsize=(8, 5))
plt.plot(power_list, np.array(ion_thrust_list) * 1e3, marker="o", label="Ion thrust (mN)")
plt.plot(power_list, np.array(total_thrust_list) * 1e3, marker="s", label="Total thrust (mN)")
plt.xlabel("RF Power (W)")
plt.ylabel("Thrust (mN)")
plt.title("Thrust vs RF Power at 250 km")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
