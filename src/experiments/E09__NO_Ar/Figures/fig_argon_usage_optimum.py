from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    OUT_DIR,
    Case,
    build_sample_from_case,
    drag_components,
    ensure_out_dir,
    isp_s,
    mdot_kg_s,
    save_json,
    solve_plasma_sample,
    weather_records,
)


# Current no-argon optimum from active_learning_until_criterion.py log.
OPT_ALTITUDE_KM = 178.52855422156537
OPT_INCLINATION_DEG = 20.0
OPT_RAAN_DEG = -13.0
OPT_AREA_M2 = 0.23586017020886357
DEFAULT_POWER_RF_W = 1000.0

TANK_MASSES_KG = np.array([0.01, 0.1, 1.0, 10.0], dtype=float)
TANK_LABELS = ["10 g", "100 g", "1 kg", "10 kg"]
M_AR = 6.63e-26


def _argon_rates(n_rates: int, rate_min: float, rate_max: float) -> np.ndarray:
    if n_rates < 2:
        return np.array([0.0], dtype=float)
    positive = np.logspace(np.log10(rate_min), np.log10(rate_max), n_rates - 1)
    return np.concatenate(([0.0], positive))


def _orbit_stats(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    area_m2: float,
    argon_rate: float,
    orbit_points: int,
    power_rf_w: float,
    fast_mode: bool,
) -> dict:
    records, f107a_map = weather_records()
    thetas = np.linspace(0.0, 2.0 * np.pi, orbit_points, endpoint=False)

    thrusts = []
    drags = []
    margins = []
    mdots = []

    for idx, theta in enumerate(thetas):
        sample = build_sample_from_case(
            Case(
                altitude_km=altitude_km,
                area_m2=area_m2,
                argon_rate=argon_rate,
                inclination_deg=inclination_deg,
                raan_deg=raan_deg,
                theta_rad=float(theta),
            ),
            records=records,
            f107a_map=f107a_map,
        )
        plasma = solve_plasma_sample(
            sample,
            power_rf_w=power_rf_w,
            fast_mode=fast_mode,
            simulation_name=f"figAr_rate_{argon_rate:.2e}_{idx:02d}",
        )
        thrust = float(plasma["thrust_final_N"])
        drag = float(drag_components(sample)["drag_total_N"])
        mdot = float(mdot_kg_s(sample))

        thrusts.append(thrust)
        drags.append(drag)
        margins.append(thrust - drag)
        mdots.append(mdot)

    thrusts_arr = np.asarray(thrusts, dtype=float)
    drags_arr = np.asarray(drags, dtype=float)
    margins_arr = np.asarray(margins, dtype=float)
    mdots_arr = np.asarray(mdots, dtype=float)
    isp_arr = np.array([isp_s(t, m) for t, m in zip(thrusts_arr, mdots_arr)], dtype=float)

    argon_mdot = float(argon_rate * M_AR)
    return {
        "argon_rate": float(argon_rate),
        "argon_mdot_kg_s": argon_mdot,
        "min_margin_N": float(np.min(margins_arr)),
        "mean_margin_N": float(np.mean(margins_arr)),
        "min_thrust_N": float(np.min(thrusts_arr)),
        "mean_thrust_N": float(np.mean(thrusts_arr)),
        "mean_drag_N": float(np.mean(drags_arr)),
        "min_isp_s": float(np.min(isp_arr)),
        "mean_isp_s": float(np.mean(isp_arr)),
        "bottleneck_index": int(np.argmin(margins_arr)),
    }


def _tank_lifetime_days(tank_mass_kg: float, argon_mdot_kg_s: float) -> float:
    if argon_mdot_kg_s <= 0.0:
        return float("inf")
    return float(tank_mass_kg / argon_mdot_kg_s / 86400.0)


def _finite_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Argon usage study around the current no-argon optimum orbit.")
    parser.add_argument("--altitude-km", type=float, default=OPT_ALTITUDE_KM)
    parser.add_argument("--inclination-deg", type=float, default=OPT_INCLINATION_DEG)
    parser.add_argument("--raan-deg", type=float, default=OPT_RAAN_DEG)
    parser.add_argument("--area-m2", type=float, default=OPT_AREA_M2)
    parser.add_argument("--power-rf-w", type=float, default=DEFAULT_POWER_RF_W)
    parser.add_argument("--orbit-points", type=int, default=12)
    parser.add_argument("--n-rates", type=int, default=7)
    parser.add_argument("--argon-rate-min", type=float, default=1e16)
    parser.add_argument("--argon-rate-max", type=float, default=5e18)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_out_dir()
    rates = _argon_rates(int(args.n_rates), float(args.argon_rate_min), float(args.argon_rate_max))
    rows = [
        _orbit_stats(
            altitude_km=float(args.altitude_km),
            inclination_deg=float(args.inclination_deg),
            raan_deg=float(args.raan_deg),
            area_m2=float(args.area_m2),
            argon_rate=float(rate),
            orbit_points=int(args.orbit_points),
            power_rf_w=float(args.power_rf_w),
            fast_mode=bool(args.fast),
        )
        for rate in rates
    ]

    baseline_margin = rows[0]["min_margin_N"]
    for row in rows:
        argon_mdot = float(row["argon_mdot_kg_s"])
        delta_margin = float(row["min_margin_N"] - baseline_margin)
        row["delta_min_margin_N"] = delta_margin
        row["gain_per_argon_kg_s"] = float(delta_margin / argon_mdot) if argon_mdot > 0.0 else float("nan")
        row["tank_lifetime_days"] = {
            label: _tank_lifetime_days(mass, argon_mdot)
            for label, mass in zip(TANK_LABELS, TANK_MASSES_KG)
        }

    positive_rows = [row for row in rows if row["argon_rate"] > 0.0]
    best_abs = max(rows, key=lambda row: row["min_margin_N"])
    best_eff = max(positive_rows, key=lambda row: row["gain_per_argon_kg_s"]) if positive_rows else None

    payload = {
        "design": {
            "altitude_km": float(args.altitude_km),
            "inclination_deg": float(args.inclination_deg),
            "raan_deg": float(args.raan_deg),
            "area_m2": float(args.area_m2),
            "power_rf_w": float(args.power_rf_w),
            "orbit_points": int(args.orbit_points),
        },
        "best_absolute_margin": best_abs,
        "best_margin_per_argon": best_eff,
        "rows": rows,
    }
    save_json(out_dir.joinpath("argon_usage_optimum.json"), payload)

    rates_arr = np.asarray([row["argon_rate"] for row in rows], dtype=float)
    min_margin_arr = np.asarray([row["min_margin_N"] for row in rows], dtype=float)
    min_thrust_arr = np.asarray([row["min_thrust_N"] for row in rows], dtype=float)
    min_isp_arr = np.asarray([row["min_isp_s"] for row in rows], dtype=float)
    eff_arr = np.asarray([row["gain_per_argon_kg_s"] for row in rows], dtype=float)

    positive_rates = _finite_positive(rates_arr)
    x_min = float(np.min(positive_rates)) if positive_rates.size else 1e17
    x_max = float(np.max(positive_rates)) if positive_rates.size else 1e18

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.semilogx(rates_arr[1:], min_margin_arr[1:], marker="o")
    ax.axhline(min_margin_arr[0], color="k", linestyle="--", alpha=0.5, label="No Ar")
    ax.set_title("Worst-case orbit margin")
    ax.set_ylabel("min(T-D) [N]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.semilogx(rates_arr[1:], min_thrust_arr[1:], marker="o", label="Min thrust")
    ax.set_title("Worst-case thrust")
    ax.set_ylabel("Thrust [N]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(rates_arr[1:], min_isp_arr[1:], marker="o")
    ax.set_title("Worst-case Isp")
    ax.set_xlabel("Argon injection rate [part/s]")
    ax.set_ylabel("Isp [s]")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if np.isfinite(eff_arr[1:]).any():
        ax.semilogx(rates_arr[1:], eff_arr[1:], marker="o")
    ax.set_title("Margin gain per argon flow")
    ax.set_xlabel("Argon injection rate [part/s]")
    ax.set_ylabel("Delta min(T-D) / m_dot_Ar [N s / kg]")
    ax.grid(True, alpha=0.3)

    for axis in axes.flat:
        axis.set_xlim(x_min, x_max)
    fig.suptitle("Argon impact around current optimum orbit")
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("argon_usage_optimum.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mass in zip(TANK_LABELS, TANK_MASSES_KG):
        lifetimes = np.asarray(
            [_tank_lifetime_days(mass, float(row["argon_mdot_kg_s"])) for row in rows[1:]],
            dtype=float,
        )
        ax.loglog(rates_arr[1:], lifetimes, marker="o", label=label)
    ax.set_xlabel("Argon injection rate [part/s]")
    ax.set_ylabel("Continuous-use lifetime [days]")
    ax.set_title("Argon tank lifetime")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("argon_tank_lifetime.png"), dpi=220)
    plt.close(fig)

    print(f"Saved argon study to {out_dir}")
    print(
        "Best absolute margin: "
        f"rate={best_abs['argon_rate']:.3e} part/s, min_margin={best_abs['min_margin_N']:.3e} N"
    )
    if best_eff is not None:
        print(
            "Best margin per argon flow: "
            f"rate={best_eff['argon_rate']:.3e} part/s, "
            f"metric={best_eff['gain_per_argon_kg_s']:.3e} N s / kg"
        )


if __name__ == "__main__":
    main()
