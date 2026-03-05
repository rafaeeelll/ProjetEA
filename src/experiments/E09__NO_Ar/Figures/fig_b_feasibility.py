from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, Case, build_sample_from_case, drag_components, isp_s, mdot_kg_s, solve_plasma_sample, weather_records


def _evaluate_grid(
    altitudes: np.ndarray,
    areas: np.ndarray,
    power_rf_w: float,
    fast_mode: bool,
) -> dict:
    records, f107a_map = weather_records()
    n_alt, n_area = len(altitudes), len(areas)
    thrust = np.full((n_alt, n_area), np.nan)
    drag = np.full((n_alt, n_area), np.nan)
    margin = np.full((n_alt, n_area), np.nan)
    isp = np.full((n_alt, n_area), np.nan)
    ratio = np.full((n_alt, n_area), np.nan)
    cache: dict[tuple[float, float], dict] = {}

    for i, alt in enumerate(altitudes):
        for j, area in enumerate(areas):
            key = (float(alt), float(area))
            if key in cache:
                sample = cache[key]["sample"]
                thrust_n = cache[key]["thrust_n"]
            else:
                sample = build_sample_from_case(
                    Case(altitude_km=float(alt), area_m2=float(area)),
                    records=records,
                    f107a_map=f107a_map,
                )
                try:
                    res = solve_plasma_sample(
                        sample=sample,
                        power_rf_w=power_rf_w,
                        fast_mode=fast_mode,
                        simulation_name=f"figB_alt{alt:.1f}_a{area:.4f}",
                    )
                    thrust_n = float(res["thrust_final_N"])
                except Exception:
                    thrust_n = 0.0
                cache[key] = {"sample": sample, "thrust_n": thrust_n}

            d = drag_components(sample)
            drag_n = float(d["drag_total_N"])
            m = thrust_n - drag_n
            mdot = mdot_kg_s(sample)
            thrust[i, j] = thrust_n
            drag[i, j] = drag_n
            margin[i, j] = m
            isp[i, j] = isp_s(thrust_n, mdot)
            ratio[i, j] = thrust_n / max(drag_n, 1e-30)

    return {
        "altitudes": altitudes,
        "areas": areas,
        "thrust_N": thrust,
        "drag_N": drag,
        "margin_N": margin,
        "isp_s": isp,
        "ratio": ratio,
    }


def figure_4_map(out_dir: Path, grid: dict) -> None:
    altitudes = grid["altitudes"]
    areas = grid["areas"]
    thrust = grid["thrust_N"]
    margin = grid["margin_N"]
    isp = grid["isp_s"]
    ratio = grid["ratio"]

    area_grid, alt_grid = np.meshgrid(areas, altitudes)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    pcm = axes[0, 0].pcolormesh(area_grid, alt_grid, np.maximum(thrust, 1e-12), norm=mcolors.LogNorm(), shading="nearest")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Thrust [N]")
    fig.colorbar(pcm, ax=axes[0, 0])

    vmax = np.nanmax(np.abs(margin))
    pcm = axes[0, 1].pcolormesh(area_grid, alt_grid, margin, cmap="RdYlGn", norm=mcolors.SymLogNorm(linthresh=1e-7, vmin=-vmax, vmax=vmax), shading="nearest")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Margin = Thrust - Drag [N]")
    fig.colorbar(pcm, ax=axes[0, 1])
    axes[0, 1].text(0.012, 245, "Upper bound: low density", fontsize=8)
    axes[0, 1].text(0.5, 155, "Lower bound: high drag", fontsize=8)
    axes[0, 1].text(0.02, 200, "Left bound: low capture", fontsize=8)
    axes[0, 1].text(0.7, 200, "Right bound: power-starved", fontsize=8)

    pcm = axes[1, 0].pcolormesh(area_grid, alt_grid, isp, cmap="plasma", shading="nearest")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Isp [s]")
    fig.colorbar(pcm, ax=axes[1, 0])

    pcm = axes[1, 1].pcolormesh(area_grid, alt_grid, ratio, cmap="viridis", shading="nearest")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("T/D ratio")
    fig.colorbar(pcm, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.set_xlabel("A_intake [m²]")
        ax.set_ylabel("Altitude [km]")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir.joinpath("B4_feasibility_map_4panels.png"), dpi=220)
    plt.close(fig)


def figure_5_cut_altitude(out_dir: Path, areas: np.ndarray, altitudes: tuple[float, float], power_rf_w: float, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, alt in zip(axes, altitudes):
        thrust_vals = []
        d_intake = []
        d_body = []
        d_lateral = []
        d_total = []
        margin = []
        for area in areas:
            sample = build_sample_from_case(Case(altitude_km=float(alt), area_m2=float(area)), records=records, f107a_map=f107a_map)
            try:
                thrust_n = solve_plasma_sample(sample, power_rf_w=power_rf_w, fast_mode=fast_mode, simulation_name=f"fig5_alt{alt:.0f}_a{area:.4f}")["thrust_final_N"]
            except Exception:
                thrust_n = 0.0
            dd = drag_components(sample, lateral_area_m2=0.0)
            thrust_vals.append(thrust_n)
            d_intake.append(dd["drag_intake_N"])
            d_body.append(dd["drag_body_N"])
            d_lateral.append(dd["drag_lateral_N"])
            d_total.append(dd["drag_total_N"])
            margin.append(thrust_n - dd["drag_total_N"])
        ax.plot(areas, thrust_vals, label="Thrust")
        ax.plot(areas, d_total, label="Drag total")
        ax.plot(areas, d_intake, "--", label="Drag intake")
        ax.plot(areas, d_body, "--", label="Drag body")
        ax.plot(areas, d_lateral, "--", label="Drag lateral (model=0)")
        ax.plot(areas, margin, linewidth=2.0, label="Margin")
        ax.set_xscale("log")
        ax.set_title(f"Altitude {alt:.0f} km")
        ax.set_xlabel("A_intake [m²]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("N")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("B5_cut_fixed_altitude.png"), dpi=220)
    plt.close(fig)


def figure_6_cut_area(out_dir: Path, altitudes: np.ndarray, areas: tuple[float, float, float], power_rf_w: float, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, area in zip(axes, areas):
        thrust_vals = []
        drag_vals = []
        margin_vals = []
        for alt in altitudes:
            sample = build_sample_from_case(Case(altitude_km=float(alt), area_m2=float(area)), records=records, f107a_map=f107a_map)
            try:
                thrust_n = solve_plasma_sample(sample, power_rf_w=power_rf_w, fast_mode=fast_mode, simulation_name=f"fig6_a{area:.3f}_alt{alt:.1f}")["thrust_final_N"]
            except Exception:
                thrust_n = 0.0
            drag_n = drag_components(sample)["drag_total_N"]
            thrust_vals.append(thrust_n)
            drag_vals.append(drag_n)
            margin_vals.append(thrust_n - drag_n)
        ax.plot(altitudes, thrust_vals, label="Thrust")
        ax.plot(altitudes, drag_vals, label="Drag")
        ax.plot(altitudes, margin_vals, label="Margin")
        ax.set_title(f"A_intake={area:.2f} m²")
        ax.set_xlabel("Altitude [km]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("N")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("B6_cut_fixed_area.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section B figures: feasibility domain.")
    parser.add_argument("--n-alt", type=int, default=9, help="Grid points in altitude for map.")
    parser.add_argument("--n-area", type=int, default=9, help="Grid points in area for map.")
    parser.add_argument("--power-rf", type=float, default=1000.0, help="RF power for thrust solves.")
    parser.add_argument("--fast", action="store_true", help="Use fast plasma solve mode.")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    altitudes = np.linspace(150.0, 250.0, int(args.n_alt))
    areas = np.logspace(np.log10(1e-2), np.log10(1.0), int(args.n_area))
    grid = _evaluate_grid(altitudes, areas, power_rf_w=float(args.power_rf), fast_mode=bool(args.fast))
    figure_4_map(out_dir, grid)
    figure_5_cut_altitude(out_dir, areas=np.logspace(np.log10(1e-2), np.log10(1.0), 16), altitudes=(180.0, 200.0), power_rf_w=float(args.power_rf), fast_mode=bool(args.fast))
    figure_6_cut_area(out_dir, altitudes=np.linspace(150.0, 250.0, 24), areas=(0.05, 0.1, 0.3), power_rf_w=float(args.power_rf), fast_mode=bool(args.fast))
    print(f"Saved Section B figures to {out_dir}")


if __name__ == "__main__":
    main()

