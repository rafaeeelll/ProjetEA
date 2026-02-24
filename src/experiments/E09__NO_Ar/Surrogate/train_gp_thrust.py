from __future__ import annotations

import pickle

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from surrogate_config import (
    GP_DATASET_PATH,
    GP_MODEL_META_PATH,
    GP_MODEL_PATH,
    MAX_TRAIN_SAMPLES,
    TRAIN_SUBSAMPLE_SEED,
)
from gp_utils import FEATURE_NAMES, read_json, write_json


def _dataset_to_arrays(path):
    data = read_json(path)
    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"Dataset empty: {path}")
    x = np.array([s["x"] for s in samples], dtype=float)
    y = np.array([s["y_total_thrust_N"] for s in samples], dtype=float)
    return x, y


def main() -> None:
    x, y = _dataset_to_arrays(GP_DATASET_PATH)
    n_total = int(x.shape[0])

    if MAX_TRAIN_SAMPLES > 0 and n_total > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(TRAIN_SUBSAMPLE_SEED)
        idx = rng.choice(n_total, size=MAX_TRAIN_SAMPLES, replace=False)
        x = x[idx]
        y = y[idx]
        print(f"Subsampled GP training set: {MAX_TRAIN_SAMPLES}/{n_total} samples (seed={TRAIN_SUBSAMPLE_SEED})")

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    xs = (x - x_mean) / x_std

    kernel = (
        ConstantKernel(1.0, (1e-6, 1e6))
        * RBF(length_scale=np.ones(xs.shape[1]), length_scale_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-14, 1e-3))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=42,
    )
    gp.fit(xs, y)

    GP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GP_MODEL_PATH.open("wb") as handle:
        pickle.dump({"gp": gp, "x_mean": x_mean, "x_std": x_std, "features": FEATURE_NAMES}, handle)

    write_json(
        GP_MODEL_META_PATH,
        {
            "model": "GaussianProcessRegressor",
            "features": FEATURE_NAMES,
            "n_samples": int(x.shape[0]),
            "n_samples_total": n_total,
            "max_train_samples": int(MAX_TRAIN_SAMPLES),
            "train_subsample_seed": int(TRAIN_SUBSAMPLE_SEED),
            "kernel_optimized": str(gp.kernel_),
            "dataset": str(GP_DATASET_PATH),
        },
    )
    print(f"Saved GP model: {GP_MODEL_PATH}")


if __name__ == "__main__":
    main()
