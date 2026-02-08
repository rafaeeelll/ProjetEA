import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.constants import e, k as k_B


INPUT_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_dataset.json")
)
OUTPUT_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")
)


def _parse_date(meta: dict) -> datetime:
    date_str = meta.get("date")
    if date_str:
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            pass
    return datetime(2020, 1, 1, 12, 0, 0)


@lru_cache(maxsize=8192)
def _msis_cached(
    altitude_km: float,
    lat_deg: float,
    lon_deg: float,
    date: datetime,
    f107: float,
    f107a: float,
    ap: float,
) -> dict:
    from nrlmsise00 import msise_model  # type: ignore

    dens, temp = msise_model(date, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
    dens = np.array(dens, dtype=float)
    temp = np.array(temp, dtype=float)

    n_N2 = float(dens[2]) * 1e6
    n_O2 = float(dens[3]) * 1e6
    n_O = float(dens[1]) * 1e6
    n_N = float(dens[7]) * 1e6
    T_K = float(temp[1])

    return {
        "N2_m3": n_N2,
        "O2_m3": n_O2,
        "O_m3": n_O,
        "N_m3": n_N,
        "T_K": T_K,
        "T_eV": float(k_B * T_K / e),
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    meta = data.get("metadata", {})
    date = _parse_date(meta)
    f107 = float(meta.get("f107", 150.0))
    f107a = float(meta.get("f107a", 150.0))
    ap = float(meta.get("ap", 4.0))

    samples = data.get("samples", [])
    if not samples:
        raise ValueError("Dataset is empty.")

    new_samples = []
    for idx, sample in enumerate(samples, start=1):
        lat = round(float(sample["lat_deg"]), 6)
        lon = round(float(sample["lon_deg"]), 6)
        alt = round(float(sample["altitude_km"]), 6)
        msis = _msis_cached(alt, lat, lon, date, f107, f107a, ap)

        enriched = dict(sample)
        enriched.update(msis)
        new_samples.append(enriched)

        if idx % 500 == 0:
            print(f"Enriched {idx}/{len(samples)} samples...")

    out = {
        "metadata": {
            **meta,
            "msis_date": date.isoformat(),
            "msis_f107": f107,
            "msis_f107a": f107a,
            "msis_ap": ap,
            "inputs": [
                "N2_m3",
                "O2_m3",
                "O_m3",
                "N_m3",
                "T_K",
                "T_eV",
                "argon_injection_rate",
            ],
            "note": "Original samples augmented with MSIS densities at fixed date.",
        },
        "samples": new_samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print(f"Wrote MSIS-augmented dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
