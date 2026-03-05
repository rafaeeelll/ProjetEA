from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.constants import e, k as k_B, pi


E09_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")
ETA_COLLECTION = 0.4  # compatibility constant for scripts that still import it
ARGON_DEFAULT_RATE = 0.0

# Keep chamber geometry fixed for surrogate consistency.
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

if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

try:
    import global_model_package  # noqa: F401
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parents[2].joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.chamber_caracteristics import Chamber
from global_model_package.model import GlobalModel
from global_model_package.reactions import ElectronHeatingConstantRFPower
from reaction_set_N_et_O import get_species_and_reactions
from Drag.drag_model import collection_efficiency


def _load_dataset(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_dataset(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _sample_to_atm(sample: dict) -> dict[str, float]:
    t_k = float(sample["T_K"])
    return {
        "N2": float(sample["N2_m3"]),
        "N": float(sample["N_m3"]),
        "O2": float(sample["O2_m3"]),
        "O": float(sample["O_m3"]),
        "T_K": t_k,
        "T_eV": float(k_B * t_k / e),
        "source": "synthetic_seed",
    }


def _thrust_for_sample(sample: dict, fast_mode: bool) -> float:
    area_m2 = float(sample["intake_area_m2"])
    altitude_km = float(sample.get("altitude_km", 200.0))
    argon_injection_rate = float(sample.get("argon_injection_rate", ARGON_DEFAULT_RATE))
    orbital_speed_m_s = sample.get("orbital_speed_m_s", None)
    if orbital_speed_m_s is not None:
        orbital_speed_m_s = float(orbital_speed_m_s)
    chamber = Chamber(CHAMBER_CONFIG)
    chamber_area_m2 = math.pi * CHAMBER_CONFIG["R"] ** 2
    eta_collection_eff = collection_efficiency(A_intake=area_m2, A_chamber=chamber_area_m2)
    atm = _sample_to_atm(sample)
    species, initial_state, reactions_list, _ = get_species_and_reactions(
        chamber=chamber,
        altitude=altitude_km,
        argon_injection_rate=argon_injection_rate,
        ion_seed=1e10,
        electron_seed=1e12,
        A_intake=area_m2,
        eta_collection=eta_collection_eff,
        orbital_speed=orbital_speed_m_s,
        atm=atm,
    )
    electron_heating = ElectronHeatingConstantRFPower(species, 1000.0, chamber)
    log_dir = PROJECT_ROOT.joinpath("outputs", "e09_dataset_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    model = GlobalModel(
        species,
        reactions_list,
        chamber,
        electron_heating,
        simulation_name="seed_dataset_thrust",
        log_folder_path=log_dir,
        fast=fast_mode,
    )
    sol = model.solve(0.0, 1e-2, initial_state)
    return float(model.total_thrust(sol.y[:, -1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute missing total_thrust_N for seed dataset samples.")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of pending samples to compute (0=all).")
    parser.add_argument("--save-every", type=int, default=20, help="Save dataset every N successful computations.")
    parser.add_argument("--fast", action="store_true", help="Enable fast solver mode.")
    args = parser.parse_args()

    payload = _load_dataset(DATASET_PATH)
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"No samples in {DATASET_PATH}")

    pending_idx = []
    for i, s in enumerate(samples):
        thrust = s.get("total_thrust_N", None)
        if thrust is None or s.get("needs_simulation", False):
            pending_idx.append(i)
    if args.max_samples > 0:
        pending_idx = pending_idx[: args.max_samples]

    print(f"Pending samples: {len(pending_idx)}")
    ok = 0
    fail = 0
    for k, idx in enumerate(pending_idx, start=1):
        s = samples[idx]
        try:
            thrust = _thrust_for_sample(s, fast_mode=bool(args.fast))
            s["total_thrust_N"] = float(thrust)
            s["needs_simulation"] = False
            ok += 1
        except Exception as exc:
            s["total_thrust_N"] = None
            s["needs_simulation"] = True
            s["simulation_error"] = str(exc)
            fail += 1

        if ok > 0 and ok % max(args.save_every, 1) == 0:
            payload["samples"] = samples
            _save_dataset(DATASET_PATH, payload)
            print(f"[{k}/{len(pending_idx)}] saved interim: ok={ok}, fail={fail}")

    payload["samples"] = samples
    meta = payload.setdefault("metadata", {})
    meta["computed_thrust_samples_ok"] = int(ok)
    meta["computed_thrust_samples_fail"] = int(fail)
    _save_dataset(DATASET_PATH, payload)
    print(f"Done. ok={ok}, fail={fail}. Updated dataset: {DATASET_PATH}")


if __name__ == "__main__":
    main()
