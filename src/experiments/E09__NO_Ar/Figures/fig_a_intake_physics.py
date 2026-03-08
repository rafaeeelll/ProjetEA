from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import Case, OUT_DIR, beta_by_species, build_sample_from_case, collection_efficiency, weather_records


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
    fig.savefig(out_dir.joinpath("A1_tradeoff_eta_beta_area_ratio.png"), dpi=200)
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
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("beta")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("A2_beta_vs_altitude_wall_temp.png"), dpi=200)
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
    fig.savefig(out_dir.joinpath("A3_atmosphere_composition_vs_altitude.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section A figures: intake physics.")
    parser.add_argument("--area-m2", type=float, default=0.1, help="Fixed intake area for figures A2/A3.")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_1_tradeoff(out_dir)
    figure_3_composition_vs_altitude(out_dir, area_m2=float(args.area_m2))
    print(f"Saved Section A figures to {out_dir}")


if __name__ == "__main__":
    main()
