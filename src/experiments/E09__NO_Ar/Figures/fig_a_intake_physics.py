from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import Case, OUT_DIR, beta_by_species, build_sample_from_case, collection_efficiency, mdot_kg_s, weather_records


def figure_1_tradeoff(out_dir: Path) -> None:
    a_chamber = np.pi * (6e-2) ** 2
    ar = np.logspace(0, 2, 120)
    a_intake = ar * a_chamber
    eta = np.array([collection_efficiency(float(a), a_chamber) for a in a_intake], dtype=float)
    u_orb = 7800.0
    betas = {sp: np.array([beta_by_species(u_orb, 300.0, float(a), float(e))[sp] for a, e in zip(a_intake, eta)], dtype=float) for sp in ["N2", "O", "O2", "N"]}

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(ar, eta, color="k", linewidth=2.0, label="eta_c")
    for sp, y in betas.items():
        ax2.plot(ar, y, label=f"beta_{sp}")
    ax1.set_xscale("log")
    ax1.set_xlabel("A_intake / A_chamber")
    ax1.set_ylabel("eta_c")
    ax2.set_ylabel("beta")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_2_6_A1_tradeoff_eta_beta_area_ratio.png"), dpi=200)
    plt.close(fig)


def figure_2_beta_vs_altitude(out_dir: Path, area_m2: float) -> None:
    records, f107a_map = weather_records()
    altitudes = np.linspace(150.0, 250.0, 60)
    wall_temps = [200.0, 300.0, 500.0]
    species = ["N2", "O", "O2", "N"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, tw in zip(axes, wall_temps):
        curves = {sp: [] for sp in species}
        for alt in altitudes:
            sample = build_sample_from_case(Case(altitude_km=float(alt), area_m2=area_m2), records=records, f107a_map=f107a_map)
            beta = beta_by_species(sample["orbital_speed_m_s"], tw, area_m2, sample["eta_collection"])
            for sp in species:
                curves[sp].append(beta[sp])
        for sp in species:
            ax.plot(altitudes, curves[sp], label=sp)
        ax.set_title(f"T_wall={tw:.0f} K")
        ax.set_xlabel("Altitude (km)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("beta")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_2_7_A2_beta_vs_altitude_wall_temp.png"), dpi=200)
    plt.close(fig)


def figure_3_composition_vs_altitude(out_dir: Path, area_m2: float) -> None:
    records, f107a_map = weather_records()
    altitudes = np.linspace(150.0, 250.0, 70)
    species = ["N2", "O", "O2", "N"]
    dens = {sp: [] for sp in species}
    dens_comp = {sp: [] for sp in species}
    for alt in altitudes:
        sample = build_sample_from_case(Case(altitude_km=float(alt), area_m2=area_m2), records=records, f107a_map=f107a_map)
        beta = beta_by_species(sample["orbital_speed_m_s"], 300.0, area_m2, sample["eta_collection"])
        for sp in species:
            key = f"{sp}_m3"
            dens[sp].append(float(sample[key]))
            dens_comp[sp].append(float(sample[key]) * beta[sp])

    fig, ax = plt.subplots(figsize=(8, 5))
    for sp in species:
        ax.plot(altitudes, dens[sp], label=f"n_{sp}")
        ax.plot(altitudes, dens_comp[sp], linestyle="--", label=f"n_{sp} * beta")
    ax.set_yscale("log")
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Density (m^-3)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_extra_A3_atmosphere_composition_vs_altitude.png"), dpi=200)
    plt.close(fig)


def figure_4_tradeoff_capture(out_dir: Path, altitude_km: float) -> None:
    records, f107a_map = weather_records()
    areas = np.logspace(np.log10(1e-2), np.log10(0.5), 120)
    eta = []
    mdot_air = []
    betas = {sp: [] for sp in ["N2", "O", "O2", "N"]}

    for area in areas:
        sample = build_sample_from_case(
            Case(altitude_km=float(altitude_km), area_m2=float(area)),
            records=records,
            f107a_map=f107a_map,
        )
        eta_eff = float(sample["eta_collection"])
        eta.append(eta_eff)
        mdot_air.append(mdot_kg_s(sample))
        beta = beta_by_species(sample["orbital_speed_m_s"], 300.0, float(area), eta_eff)
        for sp in betas:
            betas[sp].append(beta[sp])

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(areas, eta, color="k", linewidth=2.0)
    axes[0].set_ylabel(r"$\eta_c$")
    axes[0].set_title(f"Trade-off capture / compression / injection at {altitude_km:.0f} km")
    axes[0].grid(True, which="both", alpha=0.3)

    for sp, values in betas.items():
        axes[1].plot(areas, values, label=sp)
    axes[1].set_ylabel(r"$\beta$")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(ncol=4, fontsize=8, loc="best")

    axes[2].plot(areas, np.asarray(mdot_air) * 1e6, color="tab:blue", linewidth=2.0)
    axes[2].set_ylabel(r"$\dot m_{in}$ [$\mu$g/s]")
    axes[2].set_xlabel(r"$A_{intake}$ [m$^2$]")
    axes[2].set_yscale("log")
    axes[2].grid(True, which="both", alpha=0.3)

    for ax in axes:
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_2_8_A4_tradeoff_eta_beta_mdot_vs_area.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section A figures: intake physics.")
    parser.add_argument("--area-m2", type=float, default=0.24, help="Fixed intake area for figures A2/A3.")
    parser.add_argument("--tradeoff-altitude-km", type=float, default=180.0, help="Fixed altitude for the eta/beta/mdot trade-off figure.")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_1_tradeoff(out_dir)
    figure_2_beta_vs_altitude(out_dir, area_m2=float(args.area_m2))
    figure_3_composition_vs_altitude(out_dir, area_m2=float(args.area_m2))
    figure_4_tradeoff_capture(out_dir, altitude_km=float(args.tradeoff_altitude_km))
    print(f"Saved Section A figures to {out_dir}")


if __name__ == "__main__":
    main()
