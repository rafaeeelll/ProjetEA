import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator


INPUT_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")
)
OUTPUT_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_rbf_model.npz")
)
META_PATH = OUTPUT_PATH.with_suffix(".json")


RBF_SMOOTHING = 1e-6
RBF_NEIGHBORS = 128

FEATURES = [
    "log10_N2_m3",
    "log10_O2_m3",
    "log10_O_m3",
    "log10_N_m3",
    "T_K",
    "log10_argon",
]


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    samples = data.get("samples", [])
    if not samples:
        raise ValueError("Dataset is empty.")

    n2 = np.array([s["N2_m3"] for s in samples], dtype=float)
    o2 = np.array([s["O2_m3"] for s in samples], dtype=float)
    o = np.array([s["O_m3"] for s in samples], dtype=float)
    n = np.array([s["N_m3"] for s in samples], dtype=float)
    tk = np.array([s["T_K"] for s in samples], dtype=float)
    argon = np.array([s["argon_injection_rate"] for s in samples], dtype=float)
    y = np.array([s["total_thrust_N"] for s in samples], dtype=float)

    X = np.column_stack(
        [
            _safe_log10(n2),
            _safe_log10(o2),
            _safe_log10(o),
            _safe_log10(n),
            tk,
            _safe_log10(argon),
        ]
    )
    return X, y, data.get("metadata", {})


def main() -> None:
    X, y, meta = _load_dataset(INPUT_PATH)

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    Xs = (X - x_mean) / x_std

    # Build once to validate; we save the training data for inference.
    _ = RBFInterpolator(
        Xs,
        y,
        kernel="thin_plate_spline",
        smoothing=RBF_SMOOTHING,
        neighbors=RBF_NEIGHBORS,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        Xs=Xs,
        y=y,
        x_mean=x_mean,
        x_std=x_std,
        kernel="thin_plate_spline",
        smoothing=RBF_SMOOTHING,
        neighbors=RBF_NEIGHBORS,
    )

    meta_out = {
        "source_dataset": str(INPUT_PATH),
        "features": FEATURES,
        "kernel": "thin_plate_spline",
        "smoothing": RBF_SMOOTHING,
        "neighbors": RBF_NEIGHBORS,
        "n_samples": int(X.shape[0]),
        "msis": {
            "date": meta.get("msis_date", meta.get("date")),
            "f107": meta.get("msis_f107", meta.get("f107")),
            "f107a": meta.get("msis_f107a", meta.get("f107a")),
            "ap": meta.get("msis_ap", meta.get("ap")),
        },
    }
    with META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(meta_out, handle, indent=2)

    print(f"Saved model: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
