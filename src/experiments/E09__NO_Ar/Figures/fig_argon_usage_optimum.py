"""Explore ce que l'ajout d'argon change autour du point optimal.

L'idee est de regarder jusqu'ou on gagne en marge propulsive sans perdre de
vue le cout associe au debit injecte.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    OUT_DIR,
    Case,
    build_sample_from_case,
    ensure_out_dir,
    drag_components,
    isp_s,
    mdot_kg_s,
    save_json,
    solve_plasma_sample,
    weather_records,
)


OPT_ALTITUDE_KM = 178.52855422156537
OPT_INCLINATION_DEG = 20.0
OPT_RAAN_DEG = -13.0
OPT_AREA_M2 = 0.23586017020886357
DEFAULT_POWER_RF_W = 1000.0
M_AR = 6.63e-26
SECONDS_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.25


def _argon_rates(n_rates: int, rate_min: float, rate_max: float) -> np.ndarray:
    if n_rates < 2:
        return np.array([0.0], dtype=float)
    positive = np.logspace(np.log10(rate_min), np.log10(rate_max), n_rates - 1)
    return np.concatenate(([0.0], positive))


def _orbit_scan(
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

    rho_vals = []
    drag_vals = []
    thrust_vals = []
    margin_vals = []
    isp_vals = []

    for idx, theta in enumerate(thetas):
        case = Case(
            altitude_km=altitude_km,
            area_m2=area_m2,
            argon_rate=argon_rate,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            theta_rad=float(theta),
        )
        sample_dict = build_sample_from_case(case, records=records, f107a_map=f107a_map)
        plasma = solve_plasma_sample(
            sample_dict,
            power_rf_w=power_rf_w,
            fast_mode=fast_mode,
            simulation_name=f"figAr_rate_{argon_rate:.2e}_{idx:02d}",
        )
        drag = drag_components(sample_dict)
        thrust = float(plasma["thrust_final_N"])
        mdot = mdot_kg_s(sample_dict)
        rho_vals.append(float(drag["rho"]))
        drag_vals.append(float(drag["drag_total_N"]))
        thrust_vals.append(thrust)
        margin_vals.append(thrust - float(drag["drag_total_N"]))
        isp_vals.append(float(isp_s(thrust, mdot)))

    rho_arr = np.asarray(rho_vals, dtype=float)
    drag_arr = np.asarray(drag_vals, dtype=float)
    thrust_arr = np.asarray(thrust_vals, dtype=float)
    margin_arr = np.asarray(margin_vals, dtype=float)
    isp_arr = np.asarray(isp_vals, dtype=float)

    rho_min_idx = int(np.argmin(rho_arr))
    rho_max_idx = int(np.argmax(rho_arr))
    bottleneck_idx = int(np.argmin(margin_arr))
    argon_mdot = float(argon_rate * M_AR)

    return {
        "argon_rate": float(argon_rate),
        "argon_mdot_kg_s": argon_mdot,
        "argon_mdot_mg_s": float(argon_mdot * 1e6),
        "thetas_rad": thetas.tolist(),
        "rho_kg_m3": rho_arr.tolist(),
        "drag_N": drag_arr.tolist(),
        "thrust_N": thrust_arr.tolist(),
        "margin_N": margin_arr.tolist(),
        "isp_s": isp_arr.tolist(),
        "min_margin_N": float(np.min(margin_arr)),
        "mean_margin_N": float(np.mean(margin_arr)),
        "min_isp_s": float(np.min(isp_arr)),
        "mean_isp_s": float(np.mean(isp_arr)),
        "bottleneck_index": bottleneck_idx,
        "bottleneck_thrust_N": float(thrust_arr[bottleneck_idx]),
        "bottleneck_drag_N": float(drag_arr[bottleneck_idx]),
        "bottleneck_isp_s": float(isp_arr[bottleneck_idx]),
        "bottleneck_td_ratio": float(thrust_arr[bottleneck_idx] / max(drag_arr[bottleneck_idx], 1e-30)),
        "rho_min_index": rho_min_idx,
        "rho_max_index": rho_max_idx,
        "thrust_rho_min_N": float(thrust_arr[rho_min_idx]),
        "thrust_rho_max_N": float(thrust_arr[rho_max_idx]),
        "drag_rho_min_N": float(drag_arr[rho_min_idx]),
        "drag_rho_max_N": float(drag_arr[rho_max_idx]),
    }


def _zero_crossing_mg_s(rows: list[dict]) -> float | None:
    x = np.asarray([row["argon_mdot_mg_s"] for row in rows], dtype=float)
    y = np.asarray([row["min_margin_N"] for row in rows], dtype=float)
    if y.size == 0:
        return None
    if y[0] >= 0.0:
        return 0.0
    for i in range(1, len(rows)):
        if y[i] >= 0.0:
            x0 = max(x[i - 1], 1e-12)
            x1 = max(x[i], 1e-12)
            y0 = y[i - 1]
            y1 = y[i]
            if np.isclose(y1, y0):
                return float(x1)
            w = (0.0 - y0) / (y1 - y0)
            logx = np.log(x0) + w * (np.log(x1) - np.log(x0))
            return float(np.exp(logx))
    return None


def _nearest_row_by_mdot(rows: list[dict], target_mg_s: float | None) -> dict:
    positive_rows = [row for row in rows if row["argon_mdot_mg_s"] > 0.0]
    if not positive_rows:
        return rows[0]
    if target_mg_s is None or target_mg_s <= 0.0:
        return max(
            positive_rows,
            key=lambda row: float(row.get("gain_per_argon_kg_s", float("-inf"))),
        )
    return min(positive_rows, key=lambda row: abs(float(row["argon_mdot_mg_s"]) - float(target_mg_s)))


def _scenario_rows(reference_row: dict) -> list[tuple[str, float, float, float, float, float]]:
    reference_mdot = float(reference_row["argon_mdot_kg_s"])
    scenarios = [
        ("Continu pessimiste", 1.0),
        ("Variable central", 0.5),
        ("Variable optimiste", 0.3),
    ]
    out = []
    for label, chi in scenarios:
        avg_mdot = chi * reference_mdot
        avg_mg_s = avg_mdot * 1e6
        cons_g_day = avg_mdot * SECONDS_PER_DAY * 1e3
        tank_1y = avg_mdot * DAYS_PER_YEAR * SECONDS_PER_DAY
        tank_5y = 5.0 * tank_1y
        out.append((label, chi, avg_mg_s, cons_g_day, tank_1y, tank_5y))
    return out


def _want(tag: str, selected: set[str]) -> bool:
    return "all" in selected or tag in selected


def _build_altitude_comparison(
    altitude_min_km: float,
    altitude_max_km: float,
    n_altitudes: int,
    inclination_deg: float,
    raan_deg: float,
    area_m2: float,
    argon_rate: float,
    power_rf_w: float,
    fast_mode: bool,
    theta_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from common import build_sample_from_case, weather_records

    records, f107a_map = weather_records()
    altitudes = np.linspace(altitude_min_km, altitude_max_km, n_altitudes)
    thrust_air = []
    thrust_argon = []
    for alt in altitudes:
        for rate, bucket in ((0.0, thrust_air), (argon_rate, thrust_argon)):
            sample = build_sample_from_case(
                Case(
                    altitude_km=float(alt),
                    area_m2=float(area_m2),
                    argon_rate=float(rate),
                    inclination_deg=float(inclination_deg),
                    raan_deg=float(raan_deg),
                    theta_rad=float(theta_rad),
                ),
                records=records,
                f107a_map=f107a_map,
            )
            try:
                thrust = solve_plasma_sample(
                    sample,
                    power_rf_w=float(power_rf_w),
                    fast_mode=fast_mode,
                    simulation_name=f"figAr_alt_{alt:.1f}_{rate:.2e}",
                )["thrust_final_N"]
            except Exception:
                thrust = 0.0
            bucket.append(float(thrust))
    return altitudes, np.asarray(thrust_air, dtype=float), np.asarray(thrust_argon, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Argon usage study around the current no-argon optimum orbit.")
    parser.add_argument("--altitude-km", type=float, default=OPT_ALTITUDE_KM)
    parser.add_argument("--inclination-deg", type=float, default=OPT_INCLINATION_DEG)
    parser.add_argument("--raan-deg", type=float, default=OPT_RAAN_DEG)
    parser.add_argument("--area-m2", type=float, default=OPT_AREA_M2)
    parser.add_argument("--power-rf-w", type=float, default=DEFAULT_POWER_RF_W)
    parser.add_argument("--orbit-points", type=int, default=15)
    parser.add_argument("--n-rates", type=int, default=9)
    parser.add_argument("--argon-rate-min", type=float, default=1e16)
    parser.add_argument("--argon-rate-max", type=float, default=5e18)
    parser.add_argument("--altitude-compare-min-km", type=float, default=150.0)
    parser.add_argument("--altitude-compare-max-km", type=float, default=250.0)
    parser.add_argument("--altitude-compare-points", type=int, default=15)
    parser.add_argument("--theta-compare-rad", type=float, default=0.0)
    parser.add_argument(
        "--reference-argon-rate",
        type=float,
        default=None,
        help="Exact argon injection rate [part/s] used for Figure 3.15 and Table 3.1.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["3_13", "3_14", "3_15", "table_3_1", "all"],
        default=["all"],
        help="Restrict output generation to a subset of report figures.",
    )
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_out_dir()
    selected = set(args.only)
    need_reference = _want("3_15", selected) or _want("table_3_1", selected)
    need_sweep = _want("3_13", selected) or _want("3_14", selected) or (need_reference and args.reference_argon_rate is None)

    rows: list[dict]
    if need_sweep:
        rates = _argon_rates(int(args.n_rates), float(args.argon_rate_min), float(args.argon_rate_max))
        rows = [
            _orbit_scan(
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
    else:
        rows = [
            _orbit_scan(
                altitude_km=float(args.altitude_km),
                inclination_deg=float(args.inclination_deg),
                raan_deg=float(args.raan_deg),
                area_m2=float(args.area_m2),
                argon_rate=0.0,
                orbit_points=int(args.orbit_points),
                power_rf_w=float(args.power_rf_w),
                fast_mode=bool(args.fast),
            )
        ]

    baseline_margin = float(rows[0]["min_margin_N"])
    for row in rows:
        argon_mdot = float(row["argon_mdot_kg_s"])
        delta_margin = float(row["min_margin_N"] - baseline_margin)
        row["delta_min_margin_N"] = delta_margin
        row["gain_per_argon_kg_s"] = float(delta_margin / argon_mdot) if argon_mdot > 0.0 else float("nan")

    ar_min_mg_s = _zero_crossing_mg_s(rows) if len(rows) > 1 else None
    reference_row = None
    altitude_compare = thrust_air_alt = thrust_arg_alt = None
    if need_reference:
        if args.reference_argon_rate is not None:
            reference_row = _orbit_scan(
                altitude_km=float(args.altitude_km),
                inclination_deg=float(args.inclination_deg),
                raan_deg=float(args.raan_deg),
                area_m2=float(args.area_m2),
                argon_rate=float(args.reference_argon_rate),
                orbit_points=int(args.orbit_points),
                power_rf_w=float(args.power_rf_w),
                fast_mode=bool(args.fast),
            )
            reference_row["delta_min_margin_N"] = float(reference_row["min_margin_N"] - baseline_margin)
            argon_mdot = float(reference_row["argon_mdot_kg_s"])
            reference_row["gain_per_argon_kg_s"] = (
                float(reference_row["delta_min_margin_N"] / argon_mdot) if argon_mdot > 0.0 else float("nan")
            )
        else:
            reference_row = _nearest_row_by_mdot(rows, ar_min_mg_s)

        if _want("3_15", selected):
            altitude_compare, thrust_air_alt, thrust_arg_alt = _build_altitude_comparison(
                altitude_min_km=float(args.altitude_compare_min_km),
                altitude_max_km=float(args.altitude_compare_max_km),
                n_altitudes=int(args.altitude_compare_points),
                inclination_deg=float(args.inclination_deg),
                raan_deg=float(args.raan_deg),
                area_m2=float(args.area_m2),
                argon_rate=float(reference_row["argon_rate"]),
                power_rf_w=float(args.power_rf_w),
                fast_mode=bool(args.fast),
                theta_rad=float(args.theta_compare_rad),
            )

    payload = {
        "design": {
            "altitude_km": float(args.altitude_km),
            "inclination_deg": float(args.inclination_deg),
            "raan_deg": float(args.raan_deg),
            "area_m2": float(args.area_m2),
            "power_rf_w": float(args.power_rf_w),
            "orbit_points": int(args.orbit_points),
        },
        "argon_zero_crossing_mg_s": ar_min_mg_s,
        "reference_row": reference_row,
        "rows": rows,
    }
    save_json(out_dir.joinpath("argon_usage_optimum.json"), payload)

    x_mg_s = np.asarray([row["argon_mdot_mg_s"] for row in rows], dtype=float)
    thrust_rho_min_mn = 1e3 * np.asarray([row["thrust_rho_min_N"] for row in rows], dtype=float)
    thrust_rho_max_mn = 1e3 * np.asarray([row["thrust_rho_max_N"] for row in rows], dtype=float)
    drag_rho_min_mn = 1e3 * np.asarray([row["drag_rho_min_N"] for row in rows], dtype=float)
    drag_rho_max_mn = 1e3 * np.asarray([row["drag_rho_max_N"] for row in rows], dtype=float)
    min_margin_mn = 1e3 * np.asarray([row["min_margin_N"] for row in rows], dtype=float)
    min_isp_s = np.asarray([row["min_isp_s"] for row in rows], dtype=float)
    mean_isp_s = np.asarray([row["mean_isp_s"] for row in rows], dtype=float)
    bottleneck_td_ratio = np.asarray([row["bottleneck_td_ratio"] for row in rows], dtype=float)

    positive_mg_s = x_mg_s[x_mg_s > 0.0]
    linthresh = float(np.min(positive_mg_s)) if positive_mg_s.size else 1e-4
    x_plot_mg_s = positive_mg_s
    min_margin_plot_mn = min_margin_mn[x_mg_s > 0.0]
    td_ratio_plot = bottleneck_td_ratio[x_mg_s > 0.0]

    if _want("3_13", selected):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        if ar_min_mg_s is not None and ar_min_mg_s > 0.0:
            ax.axvspan(0.0, ar_min_mg_s, color="0.92", zorder=0, label="Zone non viable")
            ax.axvline(ar_min_mg_s, color="k", linestyle="--", linewidth=1.2, label=r"$\dot m_{Ar,min}$")
        ax.plot(x_mg_s, thrust_rho_min_mn, marker="o", label=r"Thrust (densité min)")
        ax.plot(x_mg_s, thrust_rho_max_mn, marker="o", label=r"Thrust (densité max)")
        ax.plot(x_mg_s, drag_rho_min_mn, linestyle="--", label=r"Drag (densité min)")
        ax.plot(x_mg_s, drag_rho_max_mn, linestyle="--", label=r"Drag (densité max)")
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlabel(r"$\dot m_{Ar}$ [mg/s]")
        ax.set_ylabel("Force [mN]")
        ax.set_title("Forces aux extrema de densité orbitale")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir.joinpath("Fig_3_13_argon_thrust_drag_vs_rate.png"), dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(x_mg_s, min_isp_s, marker="o", label="Isp min sur orbite")
        ax.plot(x_mg_s, mean_isp_s, marker="o", label="Isp moyen sur orbite")
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlabel(r"$\dot m_{Ar}$ [mg/s]")
        ax.set_ylabel("Isp [s]")
        ax.set_title("Impulsion specifique vs injection d'argon")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir.joinpath("Fig_3_13b_isp_vs_argon_rate.png"), dpi=220)
        plt.close(fig)

    if _want("3_14", selected):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        if ar_min_mg_s is not None and ar_min_mg_s > 0.0:
            ax.axvline(ar_min_mg_s, color="k", linestyle="--", linewidth=1.2, label=r"$\dot m_{Ar,min}$")
        ax.axhline(0.0, color="k", linewidth=1.0)
        ax.plot(x_plot_mg_s, min_margin_plot_mn, marker="o")
        ax.set_xscale("log")
        ax.set_xlim(left=1e-4)
        ax.set_xlabel(r"$\dot m_{Ar}$ [mg/s]")
        ax.set_ylabel(r"$\min_\theta(T-D)$ [mN]")
        ax.set_title("Marge minimale sur l'orbite")
        ax.grid(True, which="both", alpha=0.3)
        if ar_min_mg_s is not None and ar_min_mg_s > 0.0:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir.joinpath("Fig_3_14_argon_min_margin_vs_rate.png"), dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(x_plot_mg_s, td_ratio_plot, marker="o")
        ax.axhline(1.0, color="k", linewidth=1.0, linestyle="--", label="T/D = 1")
        ax.set_xscale("log")
        ax.set_xlim(left=1e-4)
        ax.set_xlabel(r"$\dot m_{Ar}$ [mg/s]")
        ax.set_ylabel("T/D au bottleneck [-]")
        ax.set_title("Rapport T/D au pire point de l'orbite")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir.joinpath("Fig_3_14b_argon_td_ratio_bottleneck.png"), dpi=220)
        plt.close(fig)

    if _want("3_15", selected):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(altitude_compare, 1e3 * thrust_air_alt, label="Air seul", linewidth=2.0)
        ax.plot(
            altitude_compare,
            1e3 * thrust_arg_alt,
            label=rf"Air + Ar ({reference_row['argon_mdot_mg_s']:.3f} mg/s)",
            linewidth=2.0,
        )
        ax.set_xlabel("Altitude [km]")
        ax.set_ylabel("Thrust [mN]")
        ax.set_title(rf"Thrust vs altitude à $A_{{intake}}={args.area_m2:.3f}$ m$^2$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir.joinpath("Fig_3_15_thrust_vs_altitude_air_vs_argon.png"), dpi=220)
        plt.close(fig)

    if _want("table_3_1", selected):
        scenario_rows = _scenario_rows(reference_row)
        fig, ax = plt.subplots(figsize=(10.0, 2.8))
        ax.axis("off")
        table = ax.table(
            cellText=[
                [
                    label,
                    f"{chi:.1f}",
                    f"{avg_mg_s:.3f}",
                    f"{cons_g_day:.2f}",
                    f"{tank_1y:.2f}",
                    f"{tank_5y:.2f}",
                ]
                for label, chi, avg_mg_s, cons_g_day, tank_1y, tank_5y in scenario_rows
            ],
            colLabels=["Scénario", "χ", "Ar moyen [mg/s]", "Conso [g/j]", "Réservoir 1 an [kg]", "Réservoir 5 ans [kg]"],
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.35)
        ax.set_title("Scénarios de consommation d'argon", pad=10)
        fig.tight_layout()
        fig.savefig(out_dir.joinpath("Table_3_1_argon_scenarios.png"), dpi=220)
        plt.close(fig)

    print(f"Saved argon report figures to {out_dir}")
    print(f"Reference argon rate: {reference_row['argon_rate']:.3e} part/s ({reference_row['argon_mdot_mg_s']:.3f} mg/s)")
    if ar_min_mg_s is not None:
        print(f"Minimum viable argon flow: {ar_min_mg_s:.3f} mg/s")


if __name__ == "__main__":
    main()
