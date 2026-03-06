from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")
MODEL_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_gp_model.pkl")
MODEL_META_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_gp_model.json")

FEATURE_NAMES = [
    "log10_N2_m3",
    "log10_O2_m3",
    "log10_O_m3",
    "log10_N_m3",
    "log10_intake_area_m2",
]


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"No samples in {path}")
    return samples


def _to_arrays(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    for s in samples:
        thrust = s.get("total_thrust_N", None)
        if thrust is None:
            continue
        if s.get("needs_simulation", False):
            continue
        x_rows.append(
            [
                float(_safe_log10(np.array([float(s["N2_m3"])]))[0]),
                float(_safe_log10(np.array([float(s["O2_m3"])]))[0]),
                float(_safe_log10(np.array([float(s["O_m3"])]))[0]),
                float(_safe_log10(np.array([float(s["N_m3"])]))[0]),
                float(_safe_log10(np.array([float(s["intake_area_m2"])]))[0]),
            ]
        )
        y_vals.append(float(thrust))
    if not x_rows:
        raise ValueError("No valid thrust samples found (all missing or pending simulation).")
    return np.array(x_rows, dtype=float), np.array(y_vals, dtype=float)


def main() -> None:
    samples = _load_dataset(DATASET_PATH)
    x, y = _to_arrays(samples)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    xs = (x - x_mean) / x_std

    kernel = (
        ConstantKernel(1.0, (1e-6, 1e6))
        * Matern(length_scale=np.ones(xs.shape[1]), length_scale_bounds=(1e-3, 1e4), nu=2.5)
        + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-12, 1e-4))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=42,
    )
    gp.fit(xs, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(
            {
                "gp": gp,
                "x_mean": x_mean,
                "x_std": x_std,
                "features": FEATURE_NAMES,
            },
            handle,
        )

    with MODEL_META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "GaussianProcessRegressor",
                "dataset": str(DATASET_PATH),
                "n_samples_used": int(x.shape[0]),
                "features": FEATURE_NAMES,
                "kernel_optimized": str(gp.kernel_),
            },
            handle,
            indent=2,
        )

    print(f"Trained GP on {x.shape[0]} samples")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {MODEL_META_PATH}")


if __name__ == "__main__":
    main()
