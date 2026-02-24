import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator


DATASET_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")
)
MODEL_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_rbf_model.npz")
)


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    return X, y


def _load_model_hyperparams(path: Path) -> tuple[str, float, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    data = np.load(path)
    return str(data["kernel"]), float(data["smoothing"]), int(data["neighbors"])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), 1e-12)
    mape = float(np.mean(np.abs(err) / denom)) * 100.0
    max_err = float(np.max(np.abs(err)))
    return {"rmse": rmse, "mae": mae, "mape_percent": mape, "max_abs": max_err}


def main() -> None:
    X, y = _load_dataset(DATASET_PATH)
    n = X.shape[0]
    if n < 3:
        raise ValueError("Dataset too small for train/validation split.")
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.8 * n)
    split = min(max(split, 1), n - 1)
    train_idx = idx[:split]
    val_idx = idx[split:]

    kernel, smoothing, neighbors = _load_model_hyperparams(MODEL_PATH)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_true = y[val_idx]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0.0] = 1.0

    Xs_train = (X_train - x_mean) / x_std
    Xs_val = (X_val - x_mean) / x_std

    rbf = RBFInterpolator(
        Xs_train,
        y_train,
        kernel=kernel,
        smoothing=smoothing,
        neighbors=neighbors,
    )

    y_pred = rbf(Xs_val)

    metrics = _metrics(y_true, y_pred)
    print("Validation metrics:", metrics)

    err = y_pred - y_true
    plt.figure(figsize=(7, 4.5))
    plt.hist(err, bins=40, alpha=0.8)
    plt.xlabel("Error (N)")
    plt.ylabel("Count")
    plt.title("Validation Error Histogram")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
