"""Trace une comparaison simple du thrust pour plusieurs aires d'intake.

C'est surtout une figure de lecture rapide pour voir comment la taille captee
deplace les niveaux de performance avec l'altitude.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, Case, build_sample_from_case, solve_plasma_sample, weather_records


AREAS_M2 = [0.05, 0.1, 0.2, 0.5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot thrust vs altitude for several intake areas.")
    parser.add_argument("--fast", action="store_true", help="Use fast plasma mode.")
    parser.add_argument("--alt-min", type=float, default=150.0, help="Minimum altitude [km].")
    parser.add_argument("--alt-max", type=float, default=250.0, help="Maximum altitude [km].")
    parser.add_argument("--n-alt", type=int, default=16, help="Number of altitude samples.")
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    records, f107a_map = weather_records()

    altitudes = np.linspace(float(args.alt_min), float(args.alt_max), int(args.n_alt))
    fig, ax = plt.subplots(figsize=(8, 5))

    for area in AREAS_M2:
        thrust_mn = []
        for alt in altitudes:
            sample = build_sample_from_case(
                Case(altitude_km=float(alt), area_m2=float(area)),
                records=records,
                f107a_map=f107a_map,
            )
            try:
                thrust_n = solve_plasma_sample(
                    sample,
                    power_rf_w=1000.0,
                    fast_mode=bool(args.fast),
                    simulation_name=f"thrust_vs_alt_{alt:.1f}_{area:.3f}",
                )["thrust_final_N"]
            except Exception:
                thrust_n = 0.0
            thrust_mn.append(1e3 * float(thrust_n))
        ax.plot(altitudes, thrust_mn, marker="o", label=fr"$A={area:.2f}\,\mathrm{{m^2}}$")

    ax.set_xlabel("Altitude [km]")
    ax.set_ylabel("Thrust [mN]")
    ax.set_title("Thrust vs altitude for several intake areas")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir.joinpath("Fig_extra_thrust_vs_altitude_multi_area.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
