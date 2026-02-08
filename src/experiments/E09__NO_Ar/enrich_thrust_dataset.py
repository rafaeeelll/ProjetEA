import json
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
from global_model_package.reactions import ElectronHeatingConstantRFPower

from reaction_set_N_et_O import get_neutral_atmosphere, get_species_and_reactions


# --- Fixed configuration (same as build_thrust_dataset.py) ---
power_w = 1000
t0, tf = 0.0, 1e-2
FAST_MODE = True

date = datetime(2020, 1, 1, 12, 0, 0)
f107 = 150.0
f107a = 150.0
ap = 4.0

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

ion_seed = 1e10
electron_seed = 1e12
compression_rate = 500
collection_rate = 0.5


# --- Enrichment grid (edit as needed) ---
altitudes_km = np.arange(170.0, 221.0, 5.0)
latitudes_deg = np.arange(-90.0, 91.0, 15.0)
longitudes_deg = np.arange(0.0, 360.0, 15.0)
argon_rates = np.logspace(16, 18, 9)


log_folder_path = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset")
)
log_folder_path.mkdir(parents=True, exist_ok=True)
output_path = log_folder_path.joinpath("thrust_dataset.json")


def _make_key(lat: float, lon: float, alt: float, argon_rate: float) -> str:
    return f"{lat:.6f}|{lon:.6f}|{alt:.3f}|{argon_rate:.6e}"


def _load_existing() -> tuple[dict, set[str]]:
    if not output_path.exists():
        return {"metadata": {}, "samples": []}, set()
    with output_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    samples = data.get("samples", [])
    keys = {
        _make_key(
            float(item["lat_deg"]),
            float(item["lon_deg"]),
            float(item["altitude_km"]),
            float(item["argon_injection_rate"]),
        )
        for item in samples
    }
    return data, keys


def _save_results(data: dict) -> None:
    tmp_path = output_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(output_path)


def main() -> None:
    results, existing_keys = _load_existing()
    results["metadata"] = {
        "date": date.isoformat(),
        "f107": f107,
        "f107a": f107a,
        "ap": ap,
        "power_w": power_w,
        "t0": t0,
        "tf": tf,
        "fast_mode": FAST_MODE,
        "config": config_dict,
        "compression_rate": compression_rate,
        "collection_rate": collection_rate,
        "ion_seed": ion_seed,
        "electron_seed": electron_seed,
        "altitudes_km": altitudes_km.tolist(),
        "latitudes_deg": latitudes_deg.tolist(),
        "longitudes_deg": longitudes_deg.tolist(),
        "argon_rates": argon_rates.tolist(),
        "note": "Enrichment run (adds only missing points).",
    }
    samples = results.setdefault("samples", [])

    chamber = Chamber(config_dict)
    total_points = (
        len(altitudes_km)
        * len(latitudes_deg)
        * len(longitudes_deg)
        * len(argon_rates)
    )
    done = 0
    skipped = 0

    for altitude_km in altitudes_km:
        for lat_deg in latitudes_deg:
            for lon_deg in longitudes_deg:
                for argon_rate in argon_rates:
                    key = _make_key(lat_deg, lon_deg, altitude_km, argon_rate)
                    if key in existing_keys:
                        skipped += 1
                        continue

                    done += 1
                    print(
                        f"[new {done} | skipped {skipped}] alt={altitude_km:.1f} km, "
                        f"lat={lat_deg:.1f}, lon={lon_deg:.1f}, argon={argon_rate:.2e}"
                    )

                    atm = get_neutral_atmosphere(
                        altitude_km,
                        lat=lat_deg,
                        lon=lon_deg,
                        date=date,
                        f107=f107,
                        f107a=f107a,
                        ap=ap,
                    )

                    species, initial_state, reactions_list, _ = get_species_and_reactions(
                        chamber,
                        altitude_km,
                        lat=lat_deg,
                        lon=lon_deg,
                        date=date,
                        ion_seed=ion_seed,
                        electron_seed=electron_seed,
                        compression_rate=compression_rate,
                        collection_rate=collection_rate,
                        argon_injection_rate=argon_rate,
                        atm=atm,
                    )

                    electron_heating = ElectronHeatingConstantRFPower(
                        species, power_w, chamber
                    )
                    model = GlobalModel(
                        species,
                        reactions_list,
                        chamber,
                        electron_heating,
                        simulation_name=(
                            f"E09_enrich_alt_{altitude_km:.0f}_lat_{lat_deg:.0f}_"
                            f"lon_{lon_deg:.0f}_argon_{argon_rate:.2e}"
                        ),
                        log_folder_path=log_folder_path,
                        fast=FAST_MODE,
                    )

                    try:
                        sol = model.solve(t0, tf, initial_state)
                    except Exception as exc:
                        _save_results(results)
                        print("Solve failed, partial dataset saved.")
                        raise exc

                    final_state = sol.y[:, -1]
                    total_thrust = float(model.total_thrust(final_state))

                    samples.append(
                        {
                            "lat_deg": float(lat_deg),
                            "lon_deg": float(lon_deg),
                            "altitude_km": float(altitude_km),
                            "argon_injection_rate": float(argon_rate),
                            "total_thrust_N": total_thrust,
                        }
                    )
                    existing_keys.add(key)
                    _save_results(results)

    print(f"Total grid points: {total_points}, new computed: {done}, skipped: {skipped}")


if __name__ == "__main__":
    main()
