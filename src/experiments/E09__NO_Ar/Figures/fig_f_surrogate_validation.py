"""Verifie que le surrogate suit correctement le modele direct.

L'idee est de comparer prediction et calcul complet sur des cas plausibles
avant de s'appuyer dessus pour l'optimisation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from common import Case, OUT_DIR, build_sample_from_case, solve_plasma_sample, weather_records
from Surrogate.train_gp_from_dataset import DATASET_PATH, _load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_gp_model.pkl")
METRICS_PATH = OUT_DIR.joinpath("Fig_4_1_gp_prediction_vs_truth_metrics.json")


def _safe_log10(x: float, floor: float = 1e-30) -> float:
    return float(np.log10(max(float(x), floor)))


def _load_gp():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing GP model: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["gp"], np.asarray(payload["x_mean"], dtype=float), np.asarray(payload["x_std"], dtype=float)


def _feature(sample: dict) -> np.ndarray:
    return np.array(
        [
            _safe_log10(sample["N2_m3"]),
            _safe_log10(sample["O2_m3"]),
            _safe_log10(sample["O_m3"]),
            _safe_log10(sample["N_m3"]),
            _safe_log10(sample["intake_area_m2"]),
        ],
        dtype=float,
    )


def _valid_samples(samples: list[dict]) -> list[dict]:
    out = []
    for s in samples:
        if s.get("total_thrust_N") is None:
            continue
        if s.get("needs_simulation", False):
            continue
        out.append(s)
    if not out:
        raise ValueError("No valid training samples found in dataset.")
    return out


def _candidate_cases(
    opt_altitude_km: float,
    opt_area_m2: float,
    alt_window_km: float,
    log_area_window: float,
    inclination_deg: float,
    raan_deg: float,
    rng: np.random.Generator,
    n_points: int,
) -> list[Case]:
    altitudes = rng.uniform(opt_altitude_km - alt_window_km, opt_altitude_km + alt_window_km, size=n_points)
    log_area_center = _safe_log10(opt_area_m2)
    log_areas = rng.uniform(log_area_center - log_area_window, log_area_center + log_area_window, size=n_points)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    return [
        Case(
            altitude_km=float(alt),
            area_m2=float(10.0 ** log_area),
            inclination_deg=float(inclination_deg),
            raan_deg=float(raan_deg),
            theta_rad=float(theta),
        )
        for alt, log_area, theta in zip(altitudes, log_areas, thetas)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 4.1: validate the current GP on new points generated near the optimum."
    )
    parser.add_argument("--n-points", type=int, default=24, help="Number of new validation points.")
    parser.add_argument("--opt-altitude-km", type=float, default=178.53, help="Reference optimum altitude [km].")
    parser.add_argument("--opt-area-m2", type=float, default=0.236, help="Reference optimum intake area [m^2].")
    parser.add_argument("--alt-window-km", type=float, default=8.0, help="Altitude half-window around the optimum [km].")
    parser.add_argument("--log-area-window", type=float, default=0.12, help="Half-window around log10(A_intake).")
    parser.add_argument("--inclination-deg", type=float, default=20.0, help="Inclination used to generate new points.")
    parser.add_argument("--raan-deg", type=float, default=-13.0, help="RAAN used to generate new points.")
    parser.add_argument("--power-rf-w", type=float, default=1000.0, help="RF power used for the true model.")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.20,
        help="Minimum Euclidean distance to the training set in normalized feature space.",
    )
    parser.add_argument("--fast", action="store_true", help="Use fast plasma mode for the true model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    gp, x_mean, x_std = _load_gp()
    records, f107a_map = weather_records()
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    train_samples = _valid_samples(_load_dataset(DATASET_PATH))
    train_features = np.vstack([_feature(s) for s in train_samples])
    train_features_norm = (train_features - x_mean) / x_std
    rng = np.random.default_rng(int(args.seed))

    y_true = []
    y_pred = []
    y_std = []
    validation_rows = []
    target_n = int(args.n_points)
    min_distance = float(args.min_distance)
    accepted = 0
    attempts = 0
    max_attempts = max(20 * target_n, 200)

    while accepted < target_n and attempts < max_attempts:
        remaining = target_n - accepted
        cases = _candidate_cases(
            opt_altitude_km=float(args.opt_altitude_km),
            opt_area_m2=float(args.opt_area_m2),
            alt_window_km=float(args.alt_window_km),
            log_area_window=float(args.log_area_window),
            inclination_deg=float(args.inclination_deg),
            raan_deg=float(args.raan_deg),
            rng=rng,
            n_points=max(remaining, 4),
        )
        for case in cases:
            if accepted >= target_n:
                break
            attempts += 1
            sample = build_sample_from_case(case, records=records, f107a_map=f107a_map)
            x = _feature(sample)
            xs = (x - x_mean) / x_std
            nearest_distance = float(np.min(np.linalg.norm(train_features_norm - xs, axis=1)))
            if nearest_distance < min_distance:
                continue

            pred, sigma = gp.predict(xs.reshape(1, -1), return_std=True)
            res = solve_plasma_sample(
                sample,
                power_rf_w=float(args.power_rf_w),
                fast_mode=bool(args.fast),
                simulation_name=f"fig41_validation_{accepted:02d}",
            )
            thrust_true = float(res["thrust_final_N"])
            thrust_pred = float(pred[0])
            sigma_pred = float(sigma[0])

            y_true.append(thrust_true)
            y_pred.append(thrust_pred)
            y_std.append(sigma_pred)
            validation_rows.append(
                {
                    "altitude_km": float(case.altitude_km),
                    "intake_area_m2": float(case.area_m2),
                    "theta_rad": float(case.theta_rad),
                    "nearest_train_distance": nearest_distance,
                    "thrust_true_N": thrust_true,
                    "thrust_pred_N": thrust_pred,
                    "sigma_gp_N": sigma_pred,
                }
            )
            accepted += 1

    if accepted < target_n:
        raise RuntimeError(
            f"Could only generate {accepted} validation points farther than {min_distance:.3f} "
            f"from the training set after {attempts} attempts. Increase the local window or lower --min-distance."
        )

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_std_arr = np.asarray(y_std, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rel_err = np.abs(y_pred_arr - y_true_arr) / np.maximum(np.abs(y_true_arr), 1e-30)
    mape = float(np.mean(rel_err))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lo = float(min(np.min(y_true_arr), np.min(y_pred_arr)))
    hi = float(max(np.max(y_true_arr), np.max(y_pred_arr)))
    pad = 0.05 * max(hi - lo, 1e-12)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    sc = ax.scatter(y_true_arr, y_pred_arr, c=y_std_arr, cmap="viridis", s=46, edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="k", linewidth=1.2, label="y = x")
    ax.set_xlabel("Thrust vrai [N]")
    ax.set_ylabel("Thrust prédit [N]")
    ax.set_title("Validation du GP sur de nouveaux points proches de l'optimum")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\sigma_{GP}$ [N]")
    ax.text(
        0.04,
        0.96,
        f"n = {accepted}\n"
        f"MAE = {mae:.2e} N\n"
        f"MAPE = {100*mape:.1f}\\%\n"
        f"$R^2$ = {r2:.3f}\n"
        f"$d_{{min}}$ = {min_distance:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.95),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR.joinpath("Fig_4_1_gp_prediction_vs_truth.png"), dpi=220)
    plt.close(fig)

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": str(MODEL_PATH),
                "n_validation_points": int(accepted),
                "n_training_samples": int(train_features.shape[0]),
                "opt_altitude_km": float(args.opt_altitude_km),
                "opt_area_m2": float(args.opt_area_m2),
                "alt_window_km": float(args.alt_window_km),
                "log_area_window": float(args.log_area_window),
                "min_distance_to_training_set": min_distance,
                "mae_N": mae,
                "mape": mape,
                "r2": r2,
                "validation_points": validation_rows,
            },
            handle,
            indent=2,
        )

    print(f"Saved figure: {OUT_DIR.joinpath('Fig_4_1_gp_prediction_vs_truth.png')}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
