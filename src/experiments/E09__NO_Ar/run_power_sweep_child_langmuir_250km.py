import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, e, pi, k as k_B


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
FAST_MODE = True
date = datetime(2020, 1, 1, 12, 0, 0)
lat = 0.0
lon = 0.0

# Grid / CL parameters (adjust gap if you have a better value)
grid_gap_m = 3e-3

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
j_cl_mix_list = []
total_thrust_list = []
ion_thrust_list = []
te_list = []
tmono_list = []
tdiato_list = []


def ion_current_density_by_species(state, species, chamber):
    n_g_tot = float(np.sum(state[: species.nb]))
    h_L = chamber.h_L(n_g_tot)
    te = state[species.nb]
    ji_by_name = {}
    for sp in species.species[1:]:
        if sp.charge == 0:
            continue
        gamma_i = chamber.gamma_ion(state[sp.index], te, sp.mass)
        ji_by_name[sp.name] = abs(sp.charge) * gamma_i * h_L
    return ji_by_name


state_seed = None
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
        fast=FAST_MODE
    )

    sol = model.solve(t0, tf, initial_state)
    final_state = sol.y[:, -1]

    ji_by_name = ion_current_density_by_species(final_state, species, chamber)
    ion_current_density = float(np.sum(list(ji_by_name.values())))
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

    # Child-Langmuir current density for ion mixture
    c_cl = (4 / 9) * epsilon_0 * (config_dict["V_grid"] ** 1.5) / (grid_gap_m**2)
    if ion_current_density > 0:
        weights = {name: val / ion_current_density for name, val in ji_by_name.items()}
        j_cl_mix = 0.0
        for name, w in weights.items():
            sp = species.get_specie_by_name(name)
            j_cl_mix += w * np.sqrt(2 * abs(sp.charge) / sp.mass)
        j_cl_mix = c_cl * j_cl_mix
    else:
        ion_sp = species.get_specie_by_name("Ar+")
        j_cl_mix = c_cl * np.sqrt(2 * abs(ion_sp.charge) / ion_sp.mass)
    j_cl_mix_list.append(float(j_cl_mix))

# Find first power where J_i >= J_CL (mixture)
power_limit = None
for p, j, jcl in zip(power_list, ion_current_density_list, j_cl_mix_list):
    if j >= jcl:
        power_limit = p
        break

print(f"Child-Langmuir limit current density (mixture, d={grid_gap_m} m): {j_cl_mix_list[-1]:.3e} A/m^2")
if power_limit is None:
    print("No power in sweep reached CL limit.")
else:
    print(f"Approx. CL-limited power: {power_limit} W")

plt.figure(figsize=(8, 5))
plt.plot(power_list, ion_current_density_list, label="Ion current density (model)")
plt.plot(power_list, j_cl_mix_list, color="red", linestyle="--", label="Child-Langmuir limit (mix)")
plt.xlabel("RF Power (W)")
plt.ylabel("Ion Current Density (A/m²)")
plt.title("Ion Current Density vs RF Power at 250 km")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Temperature vs power (electron + gas)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(power_list, te_list, label="T_e (eV)", color="blue")
ax1.set_xlabel("RF Power (W)")
ax1.set_ylabel("Electron Temperature (eV)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(power_list, np.array(tmono_list) * e / k_B, linestyle="--", label="T_mono (K)", color="red")
ax2.plot(power_list, np.array(tdiato_list) * e / k_B, linestyle="--", label="T_diato (K)", color="green")
ax2.set_ylabel("Gas Temperatures (K)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
plt.title("Temperatures vs RF Power at 250 km")
plt.tight_layout()
plt.show()

# Thrust vs power (mN)
plt.figure(figsize=(8, 5))
plt.plot(power_list, np.array(ion_thrust_list) * 1e3, label="Ion thrust (mN)")
plt.plot(power_list, np.array(total_thrust_list) * 1e3, label="Total thrust (mN)")
plt.xlabel("RF Power (W)")
plt.ylabel("Thrust (mN)")
plt.title("Thrust vs RF Power at 250 km")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
