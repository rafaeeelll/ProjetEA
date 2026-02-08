import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.constants import pi


try:
    import global_model_package  # noqa: F401
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.chamber_caracteristics import Chamber
from global_model_package.model import GlobalModel
from global_model_package.reactions import ElectronHeatingConstantRFPower, GeneralElasticCollision

from reaction_set_N_et_O import get_neutral_atmosphere, get_species_and_reactions


# --- Validation case (single-point E09 sanity check) ---
ALTITUDE_KM = 183.0
POWER_RF_W = 1000.0
ARGON_INJECTION_RATE = 1e17

DATE = datetime(2020, 1, 1, 12, 0, 0)
LAT_DEG = 0.0
LON_DEG = 0.0

ION_SEED = 1e10
ELECTRON_SEED = 1e12
COMPRESSION_RATE = 500.0
COLLECTION_RATE = 1.0

T0 = 0.0
TF = 1e-2

FAST_MODE = True
PLOT_SHOW = os.environ.get("E09_VALIDATION_SHOW", "1") == "1"
SAVE_PLOT = os.environ.get("E09_VALIDATION_SAVE_PLOT", "1") == "1"

OUTPUT_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "e09_validation")
)
OUTPUT_JSON = OUTPUT_DIR.joinpath("e09_time_series_validation.json")
OUTPUT_PNG = OUTPUT_DIR.joinpath("e09_time_series_validation.png")


def _collision_frequency(reactions_list, state: np.ndarray) -> float:
    total = 0.0
    for reaction in reactions_list:
        if isinstance(reaction, GeneralElasticCollision):
            _, freq = reaction.colliding_specie_and_collision_frequency(state)
            total += float(freq)
    return total


def _as_float_array(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=float)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    chamber = Chamber(config_dict)
    atm = get_neutral_atmosphere(ALTITUDE_KM, lat=LAT_DEG, lon=LON_DEG, date=DATE)
    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber,
        ALTITUDE_KM,
        argon_injection_rate=ARGON_INJECTION_RATE,
        lat=LAT_DEG,
        lon=LON_DEG,
        date=DATE,
        ion_seed=ION_SEED,
        electron_seed=ELECTRON_SEED,
        compression_rate=COMPRESSION_RATE,
        collection_rate=COLLECTION_RATE,
        atm=atm,
    )
    electron_heating = ElectronHeatingConstantRFPower(species, POWER_RF_W, chamber)

    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name="e09_time_series_validation",
        log_folder_path=OUTPUT_DIR,
        fast=FAST_MODE,
    )

    print("Running E09 validation simulation...")
    sol = model.solve(T0, TF, initial_state)
    print(f"Solve finished: success={sol.success}, status={sol.status}, nfev={sol.nfev}")

    t = np.asarray(sol.t, dtype=float)
    y = np.asarray(sol.y, dtype=float).T

    species_names = [sp.name for sp in species.species]
    densities = y[:, : species.nb]
    te = y[:, species.nb]
    tmono = y[:, species.nb + 1]
    tdiato = y[:, species.nb + 2]

    ion_thrust = []
    neutral_thrust = []
    total_thrust = []
    ion_current = []
    collision_freq = []
    absorbed_power = []
    ion_density_sum = []

    for ti, state in zip(t, y, strict=False):
        ion_thrust.append(model.total_ion_thrust(state))
        neutral_thrust.append(model.total_neutral_thrust(state))
        total_thrust.append(model.total_thrust(state))
        ion_current.append(model.total_ion_current(state))

        nu = _collision_frequency(reactions_list, state)
        collision_freq.append(nu)
        absorbed_power.append(electron_heating.absorbed_power(state, nu, ti))

        ion_density_sum.append(sum(state[sp.index] for sp in species.species if sp.charge > 0))

    ion_thrust = _as_float_array(ion_thrust)
    neutral_thrust = _as_float_array(neutral_thrust)
    total_thrust = _as_float_array(total_thrust)
    ion_current = _as_float_array(ion_current)
    collision_freq = _as_float_array(collision_freq)
    absorbed_power = _as_float_array(absorbed_power)
    ion_density_sum = _as_float_array(ion_density_sum)

    ne = densities[:, 0]
    quasi_neutral_rel = np.abs(ne - ion_density_sum) / np.maximum(np.abs(ion_density_sum), 1e-30)

    if hasattr(model, "_compute_dy"):
        dy_final = model._compute_dy(float(t[-1]), y[-1], track_values=False)
        span = max(TF - T0, 1e-12)
        scale = np.maximum(np.abs(y[-1]), 1e-30)
        final_rel_change_over_span = float(np.max(np.abs(dy_final) / scale) * span)
    else:
        final_rel_change_over_span = float("nan")

    checks = {
        "solver_success": bool(sol.success),
        "solver_status": int(sol.status),
        "solver_message": str(sol.message),
        "nfev": int(sol.nfev),
        "njev": int(sol.njev),
        "nlu": int(sol.nlu),
        "n_time_points": int(len(t)),
        "has_nan_state": bool(np.isnan(y).any()),
        "has_inf_state": bool(np.isinf(y).any()),
        "min_density_m3": float(np.min(densities)),
        "min_temperature_eV": float(np.min(y[:, species.nb:])),
        "max_quasi_neutrality_rel_error": float(np.max(quasi_neutral_rel)),
        "final_rel_change_over_span": final_rel_change_over_span,
        "final_total_thrust_N": float(total_thrust[-1]),
        "final_ion_thrust_N": float(ion_thrust[-1]),
        "final_neutral_thrust_N": float(neutral_thrust[-1]),
        "final_ion_current_A_m2_like": float(ion_current[-1]),
        "final_absorbed_power_W": float(absorbed_power[-1]),
    }

    print("\nValidation summary:")
    for key, value in checks.items():
        print(f"  {key}: {value}")

    final_state = {
        "densities_m3": {sp.name: float(y[-1, sp.index]) for sp in species.species},
        "T_e_eV": float(te[-1]),
        "T_mono_eV": float(tmono[-1]),
        "T_diato_eV": float(tdiato[-1]),
    }

    payload = {
        "config": {
            "altitude_km": ALTITUDE_KM,
            "power_rf_W": POWER_RF_W,
            "argon_injection_rate": ARGON_INJECTION_RATE,
            "date": DATE.isoformat(),
            "lat_deg": LAT_DEG,
            "lon_deg": LON_DEG,
            "ion_seed": ION_SEED,
            "electron_seed": ELECTRON_SEED,
            "compression_rate": COMPRESSION_RATE,
            "collection_rate": COLLECTION_RATE,
            "t0_s": T0,
            "tf_s": TF,
        },
        "checks": checks,
        "final_state": final_state,
        "time_s": t.tolist(),
        "time_series": {
            "densities_m3": {sp.name: y[:, sp.index].tolist() for sp in species.species},
            "T_e_eV": te.tolist(),
            "T_mono_eV": tmono.tolist(),
            "T_diato_eV": tdiato.tolist(),
            "ion_thrust_N": ion_thrust.tolist(),
            "neutral_thrust_N": neutral_thrust.tolist(),
            "total_thrust_N": total_thrust.tolist(),
            "ion_current": ion_current.tolist(),
            "collision_frequency_s-1": collision_freq.tolist(),
            "absorbed_power_W": absorbed_power.tolist(),
            "quasi_neutral_rel_error": quasi_neutral_rel.tolist(),
        },
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved validation data: {OUTPUT_JSON}")

    if SAVE_PLOT or PLOT_SHOW:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        ax = axes[0, 0]
        for i, name in enumerate(species_names):
            ax.plot(t, densities[:, i], label=name)
        ax.set_yscale("log")
        ax.set_ylabel("Density (m$^{-3}$)")
        ax.set_title("Species Densities")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)

        ax = axes[0, 1]
        ax.plot(t, te, label="T_e")
        ax.plot(t, tmono, label="T_mono")
        ax.plot(t, tdiato, label="T_diato")
        ax.set_ylabel("Temperature (eV)")
        ax.set_title("Temperatures")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        ax.plot(t, ion_thrust, label="Ion thrust")
        ax.plot(t, neutral_thrust, label="Neutral thrust")
        ax.plot(t, total_thrust, label="Total thrust", linewidth=2)
        ax.plot(t, ion_current, label="Ion current")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("N / current")
        ax.set_title("Thrust and Ion Current")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(t, absorbed_power, label="Absorbed power (W)")
        ax2 = ax.twinx()
        ax2.plot(t, quasi_neutral_rel, color="tab:red", linestyle="--", label="Quasi-neutral rel. err")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax2.set_ylabel("Relative error")
        ax.set_title("Power and Quasi-neutrality")
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

        fig.suptitle("E09 Validation at Fixed RF Power", fontsize=14)
        fig.tight_layout()
        if SAVE_PLOT:
            fig.savefig(OUTPUT_PNG, dpi=200)
            print(f"Saved validation plot: {OUTPUT_PNG}")
        if PLOT_SHOW:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
