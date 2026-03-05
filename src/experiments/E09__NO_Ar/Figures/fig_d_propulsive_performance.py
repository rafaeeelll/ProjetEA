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
    ion_thrust_breakdown,
    isp_s,
    mdot_kg_s,
    solve_plasma_sample,
    weather_records,
)


def figure_11_isp_vs_altitude(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    altitudes = np.linspace(150.0, 250.0, 18)
    areas = [0.05, 0.1, 0.2, 0.5]
    fig, ax = plt.subplots(figsize=(8, 5))
    for area in areas:
        isp_curve = []
        for alt in altitudes:
            sample = build_sample_from_case(Case(altitude_km=float(alt), area_m2=float(area)), records=records, f107a_map=f107a_map)
            try:
                thrust = solve_plasma_sample(sample, power_rf_w=1000.0, fast_mode=fast_mode, simulation_name=f"fig11_{alt:.1f}_{area:.3f}")["thrust_final_N"]
            except Exception:
                thrust = 0.0
            isp_curve.append(isp_s(thrust, mdot_kg_s(sample)))
        ax.plot(altitudes, isp_curve, label=f"A={area:.2f} m²")
    ax.axhline(6.0, linestyle="--", color="k", label="Isp thermique ~6 s")
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Isp [s]")
    ax.set_title("Isp vs altitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("D11_isp_vs_altitude.png"), dpi=220)
    plt.close(fig)


def figure_12_thrust_decomposition(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    altitudes = np.linspace(160.0, 240.0, 10)
    ions = ["N2+", "O+", "N+", "O2+", "neutral"]
    stack = {k: [] for k in ions}
    for alt in altitudes:
        sample = build_sample_from_case(Case(altitude_km=float(alt), area_m2=0.1), records=records, f107a_map=f107a_map)
        res = solve_plasma_sample(sample, power_rf_w=1000.0, fast_mode=fast_mode, simulation_name=f"fig12_{alt:.1f}")
        contrib = ion_thrust_breakdown(res["model"], res["species"], res["states"][-1])
        for k in ions:
            stack[k].append(contrib.get(k, 0.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros_like(altitudes, dtype=float)
    for k in ions:
        vals = np.asarray(stack[k], dtype=float)
        ax.bar(altitudes, vals, bottom=bottom, width=6.0, label=k)
        bottom += vals
    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Thrust contribution [N]")
    ax.set_title("Thrust decomposition by species")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("D12_thrust_decomposition.png"), dpi=220)
    plt.close(fig)


def figure_13_td_contours_vs_power(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    p_rf_values = [500.0, 1000.0, 2000.0, 5000.0]
    altitudes = np.linspace(160.0, 240.0, 7)
    areas = np.logspace(np.log10(1e-2), np.log10(0.8), 7)
    area_grid, alt_grid = np.meshgrid(areas, altitudes)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, p_rf in zip(axes.ravel(), p_rf_values):
        ratio = np.full((len(altitudes), len(areas)), np.nan)
        for i, alt in enumerate(altitudes):
            for j, area in enumerate(areas):
                sample = build_sample_from_case(Case(altitude_km=float(alt), area_m2=float(area)), records=records, f107a_map=f107a_map)
                try:
                    thrust = solve_plasma_sample(sample, power_rf_w=p_rf, fast_mode=fast_mode, simulation_name=f"fig13_{int(p_rf)}_{alt:.1f}_{area:.4f}")["thrust_final_N"]
                except Exception:
                    thrust = 0.0
                drag = drag_components(sample)["drag_total_N"]
                ratio[i, j] = thrust / max(drag, 1e-30)
        pcm = ax.pcolormesh(area_grid, alt_grid, ratio, shading="nearest", cmap="viridis")
        ax.set_xscale("log")
        ax.set_title(f"P_RF={p_rf:.0f} W")
        fig.colorbar(pcm, ax=ax)
        ax.grid(True, alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("A_intake [m²]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Altitude [km]")
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("D13_td_ratio_iso_prf.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section D figures: propulsive performance.")
    parser.add_argument("--fast", action="store_true", help="Use fast plasma mode.")
    args = parser.parse_args()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_11_isp_vs_altitude(out_dir, fast_mode=bool(args.fast))
    figure_12_thrust_decomposition(out_dir, fast_mode=bool(args.fast))
    figure_13_td_contours_vs_power(out_dir, fast_mode=bool(args.fast))
    print(f"Saved Section D figures to {out_dir}")


if __name__ == "__main__":
    main()

