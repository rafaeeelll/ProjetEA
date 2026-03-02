from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from compute_thrust_for_dataset import ETA_COLLECTION, _thrust_for_sample


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")
MODEL_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_gp_model.pkl")
LOG_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "active_learning_log.json")

DATE_REF = datetime(2020, 1, 1, 12, 0, 0)
ALTITUDE_BOUNDS_KM = (170.0, 240.0)
INCLINATION_BOUNDS_DEG = (0.0, 98.0)
RAAN_BOUNDS_DEG = (-180.0, 180.0)
LOG10_AREA_BOUNDS = (np.log10(1e-2), np.log10(0.5))

EARTH_RADIUS_M = 6371e3
EARTH_MU = 3.986004418e14
EARTH_OMEGA = 7.2921159e-5

M_N2 = 4.65e-26
M_O2 = 5.31e-26
M_O = 2.67e-26
M_N = 2.33e-26


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _sample_to_feature(s: dict) -> np.ndarray:
    return np.array(
        [
            float(_safe_log10(np.array([float(s["N2_m3"])]))[0]),
            float(_safe_log10(np.array([float(s["O2_m3"])]))[0]),
            float(_safe_log10(np.array([float(s["O_m3"])]))[0]),
            float(_safe_log10(np.array([float(s["N_m3"])]))[0]),
            float(s["T_K"]),
            float(_safe_log10(np.array([float(s["intake_area_m2"])]))[0]),
        ],
        dtype=float,
    )


def _load_dataset() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_dataset(payload: dict) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing GP model: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["gp"], np.asarray(payload["x_mean"], dtype=float), np.asarray(payload["x_std"], dtype=float)


def _train_model() -> None:
    from train_gp_from_dataset import main as train_main

    train_main()


def _train_arrays(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    for s in samples:
        thrust = s.get("total_thrust_N", None)
        if thrust is None:
            continue
        if s.get("needs_simulation", False):
            continue
        x_rows.append(_sample_to_feature(s))
        y_vals.append(float(thrust))
    if not x_rows:
        raise ValueError("No computed training sample in dataset.")
    return np.array(x_rows, dtype=float), np.array(y_vals, dtype=float)


def _orbit_state(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    theta_rad: float,
    date_ref: datetime,
) -> tuple[datetime, float, float, float]:
    radius = EARTH_RADIUS_M + altitude_km * 1000.0
    mean_motion = np.sqrt(EARTH_MU / radius**3)
    orbital_speed_m_s = np.sqrt(EARTH_MU / radius)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    t = theta_rad / mean_motion
    dt = date_ref + timedelta(seconds=float(t))

    x_orb = radius * np.cos(theta_rad)
    y_orb = radius * np.sin(theta_rad)

    x_inc = x_orb
    y_inc = y_orb * np.cos(inc)
    z_inc = y_orb * np.sin(inc)

    x_eci = x_inc * np.cos(raan) - y_inc * np.sin(raan)
    y_eci = x_inc * np.sin(raan) + y_inc * np.cos(raan)
    z_eci = z_inc

    cos_e = np.cos(EARTH_OMEGA * t)
    sin_e = np.sin(EARTH_OMEGA * t)
    x_ecef = x_eci * cos_e + y_eci * sin_e
    y_ecef = -x_eci * sin_e + y_eci * cos_e
    z_ecef = z_eci

    lat_deg = float(np.degrees(np.arcsin(z_ecef / radius)))
    lon_deg = float(np.degrees(np.arctan2(y_ecef, x_ecef)))
    return dt, lat_deg, lon_deg, float(orbital_speed_m_s)


def _mass_density(sample: dict) -> float:
    return (
        float(sample["N2_m3"]) * M_N2
        + float(sample["O2_m3"]) * M_O2
        + float(sample["O_m3"]) * M_O
        + float(sample["N_m3"]) * M_N
    )


def _drag_newton(sample: dict, cd: float) -> float:
    rho = _mass_density(sample)
    u = float(sample["orbital_speed_m_s"])
    a = float(sample["intake_area_m2"])
    return 0.5 * rho * u * u * float(cd) * a


def _design_to_orbit_samples(
    design: np.ndarray,
    date_ref: datetime,
    records: dict,
    f107a_map: dict,
    points_per_orbit: int,
) -> list[dict]:
    from nrlmsise00 import msise_model  # type: ignore
    from msis_densities import _space_weather_params

    altitude_km = float(design[0])
    inclination_deg = float(design[1])
    raan_deg = float(design[2])
    area_m2 = float(10 ** float(design[3]))

    thetas = np.linspace(0.0, 2.0 * np.pi, int(points_per_orbit), endpoint=False)
    samples: list[dict] = []
    for theta in thetas:
        dt, lat_deg, lon_deg, orbital_speed = _orbit_state(
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            theta_rad=float(theta),
            date_ref=date_ref,
        )
        f107, f107a, ap = _space_weather_params(dt, records, f107a_map)
        dens, temp = msise_model(dt, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
        dens = np.asarray(dens, dtype=float)
        samples.append(
            {
                "N2_m3": float(dens[2]) * 1e6,
                "O2_m3": float(dens[3]) * 1e6,
                "O_m3": float(dens[1]) * 1e6,
                "N_m3": float(dens[7]) * 1e6,
                "T_K": float(temp[1]),
                "intake_area_m2": area_m2,
                "altitude_km": altitude_km,
                "orbital_speed_m_s": orbital_speed,
                "inclination_deg": inclination_deg,
                "raan_deg": raan_deg,
                "theta_rad": float(theta),
                "lat_deg": float(lat_deg),
                "lon_deg": float(lon_deg),
                "date": dt.isoformat(),
                "f107": float(f107),
                "f107a": float(f107a),
                "ap": float(ap),
                "eta_collection": float(ETA_COLLECTION),
                "total_thrust_N": None,
                "needs_simulation": True,
            }
        )
    return samples


def _predict_design_stats(
    design: np.ndarray,
    gp,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    beta_lcb: float,
    cd: float,
    date_ref: datetime,
    records: dict,
    f107a_map: dict,
    points_per_orbit: int,
) -> dict:
    samples = _design_to_orbit_samples(
        design=design,
        date_ref=date_ref,
        records=records,
        f107a_map=f107a_map,
        points_per_orbit=points_per_orbit,
    )
    x = np.array([_sample_to_feature(s) for s in samples], dtype=float)
    xs = (x - x_mean) / x_std
    mu, sigma = gp.predict(xs, return_std=True)
    drag = np.array([_drag_newton(s, cd=cd) for s in samples], dtype=float)
    margin_mu = mu - drag
    margin_lcb = margin_mu - float(beta_lcb) * sigma
    idx = int(np.argmin(margin_lcb))
    thrust_mu_min = float(np.min(mu))
    return {
        "samples": samples,
        "mu": mu,
        "sigma": sigma,
        "drag": drag,
        "margin_mu": margin_mu,
        "margin_lcb": margin_lcb,
        "j_mu_N": float(np.min(margin_mu)),
        "j_lcb_N": float(np.min(margin_lcb)),
        "idx_bottleneck": idx,
        "sigma_bottleneck_N": float(sigma[idx]),
        "thrust_mu_bottleneck_N": float(mu[idx]),
        "thrust_mu_min_N": thrust_mu_min,
    }


def _optimize_design(
    gp,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    beta_lcb: float,
    cd: float,
    date_ref: datetime,
    records: dict,
    f107a_map: dict,
    points_per_orbit: int,
    maxiter: int,
) -> tuple[np.ndarray, dict]:
    bounds = [
        (ALTITUDE_BOUNDS_KM[0], ALTITUDE_BOUNDS_KM[1]),
        (INCLINATION_BOUNDS_DEG[0], INCLINATION_BOUNDS_DEG[1]),
        (RAAN_BOUNDS_DEG[0], RAAN_BOUNDS_DEG[1]),
        (LOG10_AREA_BOUNDS[0], LOG10_AREA_BOUNDS[1]),
    ]

    def objective(z: np.ndarray) -> float:
        stats = _predict_design_stats(
            design=np.asarray(z, dtype=float),
            gp=gp,
            x_mean=x_mean,
            x_std=x_std,
            beta_lcb=beta_lcb,
            cd=cd,
            date_ref=date_ref,
            records=records,
            f107a_map=f107a_map,
            points_per_orbit=points_per_orbit,
        )
        return -float(stats["j_lcb_N"])

    res = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max(1, int(maxiter)),
        popsize=8,
        tol=1e-4,
        seed=42,
        polish=True,
        updating="deferred",
        workers=1,
    )
    design_star = np.asarray(res.x, dtype=float)
    stats_star = _predict_design_stats(
        design=design_star,
        gp=gp,
        x_mean=x_mean,
        x_std=x_std,
        beta_lcb=beta_lcb,
        cd=cd,
        date_ref=date_ref,
        records=records,
        f107a_map=f107a_map,
        points_per_orbit=points_per_orbit,
    )
    return design_star, stats_star


def _evaluate_true_design_min_margin(
    design: np.ndarray,
    cd: float,
    fast_mode: bool,
    date_ref: datetime,
    records: dict,
    f107a_map: dict,
    points_per_orbit: int,
) -> dict:
    samples = _design_to_orbit_samples(
        design=design,
        date_ref=date_ref,
        records=records,
        f107a_map=f107a_map,
        points_per_orbit=points_per_orbit,
    )
    margins: list[float] = []
    thrusts: list[float] = []
    for s in samples:
        thrust = float(_thrust_for_sample(s, fast_mode=fast_mode))
        drag = float(_drag_newton(s, cd=cd))
        margins.append(thrust - drag)
        thrusts.append(thrust)
    margins_arr = np.asarray(margins, dtype=float)
    idx = int(np.argmin(margins_arr))
    return {
        "true_min_margin_N": float(np.min(margins_arr)),
        "true_min_margin_idx": idx,
        "true_min_margin_theta_rad": float(samples[idx]["theta_rad"]),
        "true_thrust_at_min_margin_N": float(thrusts[idx]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Coupled active learning + orbit optimization (max-min thrust-drag).")
    parser.add_argument("--rel-tol", type=float, default=0.1, help="Uncertainty stop: sigma_bottleneck <= rel_tol * |thrust_mu_min|.")
    parser.add_argument("--max-iters", type=int, default=80, help="Maximum BO iterations.")
    parser.add_argument("--check-every", type=int, default=5, help="Convergence check period.")
    parser.add_argument("--patience", type=int, default=3, help="Number of check windows without significant improvement.")
    parser.add_argument("--improve-tol", type=float, default=1e-6, help="Minimum robust-objective improvement in N to reset patience.")
    parser.add_argument("--beta-lcb", type=float, default=2.0, help="LCB exploration weight for robust objective.")
    parser.add_argument("--cd", type=float, default=2.2, help="Front drag coefficient.")
    parser.add_argument("--orbit-points", type=int, default=12, help="Points used on each orbit for objective evaluation.")
    parser.add_argument("--design-maxiter", type=int, default=20, help="Inner differential-evolution maxiter.")
    parser.add_argument("--fast", action="store_true", help="Use fast mode for true 0D solves.")
    parser.add_argument("--final-verify", action="store_true", help="Run a final true full-orbit min-margin verification.")
    args = parser.parse_args()

    from msis_densities import _compute_f107a, _parse_space_weather, SPACE_WEATHER_PATH

    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)

    best_j_lcb = -np.inf
    best_design = None
    no_improve_checks = 0
    log_rows: list[dict] = []

    for it in range(1, int(args.max_iters) + 1):
        _train_model()
        gp, x_mean, x_std = _load_model()
        payload = _load_dataset()
        samples = payload.get("samples", [])
        x_train, y_train = _train_arrays(samples)

        design_star, stats = _optimize_design(
            gp=gp,
            x_mean=x_mean,
            x_std=x_std,
            beta_lcb=float(args.beta_lcb),
            cd=float(args.cd),
            date_ref=DATE_REF,
            records=records,
            f107a_map=f107a_map,
            points_per_orbit=int(args.orbit_points),
            maxiter=int(args.design_maxiter),
        )

        idx_b = int(stats["idx_bottleneck"])
        bottleneck = dict(stats["samples"][idx_b])
        bottleneck["selected_by_bo_iteration"] = int(it)
        bottleneck["design_altitude_km"] = float(design_star[0])
        bottleneck["design_inclination_deg"] = float(design_star[1])
        bottleneck["design_raan_deg"] = float(design_star[2])
        bottleneck["design_intake_area_m2"] = float(10 ** float(design_star[3]))

        try:
            true_thrust = float(_thrust_for_sample(bottleneck, fast_mode=bool(args.fast)))
            bottleneck["total_thrust_N"] = true_thrust
            bottleneck["needs_simulation"] = False
        except Exception as exc:
            true_thrust = 0.0
            bottleneck["total_thrust_N"] = 0.0
            bottleneck["needs_simulation"] = False
            bottleneck["simulation_error"] = str(exc)

        samples.append(bottleneck)
        payload["samples"] = samples
        _save_dataset(payload)

        j_lcb = float(stats["j_lcb_N"])
        improved = j_lcb > best_j_lcb + float(args.improve_tol)
        if improved:
            best_j_lcb = j_lcb
            best_design = design_star.copy()

        check_now = (it == 1) or (it % max(1, int(args.check_every)) == 0)
        sigma_target = float(args.rel_tol) * max(abs(float(stats["thrust_mu_min_N"])), 1e-12)
        sigma_ok = float(stats["sigma_bottleneck_N"]) <= sigma_target

        if check_now:
            if improved:
                no_improve_checks = 0
            else:
                no_improve_checks += 1

        row = {
            "iteration": int(it),
            "train_count": int(len(y_train)),
            "design_altitude_km": float(design_star[0]),
            "design_inclination_deg": float(design_star[1]),
            "design_raan_deg": float(design_star[2]),
            "design_intake_area_m2": float(10 ** float(design_star[3])),
            "j_mu_N": float(stats["j_mu_N"]),
            "j_lcb_N": j_lcb,
            "sigma_bottleneck_N": float(stats["sigma_bottleneck_N"]),
            "sigma_target_N": float(sigma_target),
            "sigma_ok": bool(sigma_ok),
            "true_thrust_bottleneck_N": float(true_thrust),
            "improved_best_lcb": bool(improved),
            "best_j_lcb_so_far_N": float(best_j_lcb),
            "check_now": bool(check_now),
            "no_improve_checks": int(no_improve_checks),
        }
        log_rows.append(row)
        print(
            f"[{it}] J_lcb={j_lcb:.3e} N, J_mu={float(stats['j_mu_N']):.3e} N, "
            f"sigma_b={float(stats['sigma_bottleneck_N']):.3e}, best={best_j_lcb:.3e} N"
        )

        if check_now and (no_improve_checks >= int(args.patience)) and sigma_ok:
            print(
                f"Stopping: converged (no_improve_checks={no_improve_checks}, "
                f"sigma_bottleneck={float(stats['sigma_bottleneck_N']):.3e} <= {sigma_target:.3e})"
            )
            break

    final_summary = {}
    if best_design is not None and bool(args.final_verify):
        verify = _evaluate_true_design_min_margin(
            design=best_design,
            cd=float(args.cd),
            fast_mode=bool(args.fast),
            date_ref=DATE_REF,
            records=records,
            f107a_map=f107a_map,
            points_per_orbit=int(args.orbit_points),
        )
        final_summary["best_design"] = {
            "altitude_km": float(best_design[0]),
            "inclination_deg": float(best_design[1]),
            "raan_deg": float(best_design[2]),
            "intake_area_m2": float(10 ** float(best_design[3])),
        }
        final_summary.update(verify)
        print(
            "Final true verification: "
            f"min(thrust-drag)={verify['true_min_margin_N']:.3e} N at theta={verify['true_min_margin_theta_rad']:.3f} rad"
        )

    with LOG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "settings": {
                    "rel_tol": float(args.rel_tol),
                    "max_iters": int(args.max_iters),
                    "check_every": int(args.check_every),
                    "patience": int(args.patience),
                    "improve_tol_N": float(args.improve_tol),
                    "beta_lcb": float(args.beta_lcb),
                    "cd": float(args.cd),
                    "orbit_points": int(args.orbit_points),
                    "design_maxiter": int(args.design_maxiter),
                    "eta_collection": float(ETA_COLLECTION),
                },
                "iterations": log_rows,
                "final_summary": final_summary,
            },
            handle,
            indent=2,
        )
    print(f"Saved optimization log: {LOG_PATH}")


if __name__ == "__main__":
    main()
