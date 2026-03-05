# plot_ignition_map.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from Surrogate.compute_thrust_for_dataset import ETA_COLLECTION, _thrust_for_sample
from Drag.drag_model import (
    A_BODY_M2_DEFAULT,
    CD_BODY_DEFAULT,
    drag_total_fmf,
    mass_density_from_number_densities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

EARTH_RADIUS_M = 6371e3
EARTH_MU = 3.986004418e14

DATE_REF = datetime(2020, 1, 1, 12, 0, 0)

# ── Drag (même modèle que active_learning_orbit) ──
CD_BODY = CD_BODY_DEFAULT
A_BODY_M2 = A_BODY_M2_DEFAULT


def scan_ignition_map(
    altitude_range_km: np.ndarray,
    area_range_m2: np.ndarray,
    fast_mode: bool = True,
    date_ref: datetime = DATE_REF,
    lat: float = 0.0,
    lon: float = 0.0,
) -> dict:
    """
    Scan complet 2D : pour chaque (altitude, area), calcule thrust, drag, Isp.
    """
    from msis_densities import (
        _compute_f107a,
        _parse_space_weather,
        _space_weather_params,
        SPACE_WEATHER_PATH,
    )
    from nrlmsise00 import msise_model

    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)

    n_alt = len(altitude_range_km)
    n_area = len(area_range_m2)

    thrust_map = np.full((n_alt, n_area), np.nan)
    drag_map = np.full((n_alt, n_area), np.nan)
    margin_map = np.full((n_alt, n_area), np.nan)
    isp_map = np.full((n_alt, n_area), np.nan)
    mdot_map = np.full((n_alt, n_area), np.nan)
    eta_ion_map = np.full((n_alt, n_area), np.nan)

    total = n_alt * n_area
    count = 0

    for i, alt in enumerate(altitude_range_km):
        r_orbit = EARTH_RADIUS_M + alt * 1e3
        u_orb = np.sqrt(EARTH_MU / r_orbit)
        f107, f107a, ap = _space_weather_params(date_ref, records, f107a_map)
        dens, temp = msise_model(date_ref, alt, lat, lon, f107a, f107, ap)
        dens = np.asarray(dens, dtype=float)

        n2 = float(dens[2]) * 1e6
        o2 = float(dens[3]) * 1e6
        o_at = float(dens[1]) * 1e6
        n_at = float(dens[7]) * 1e6
        T_K = float(temp[1])

        rho = mass_density_from_number_densities(n2, o2, o_at, n_at)

        for j, area in enumerate(area_range_m2):
            count += 1
            print(
                f"  [{count}/{total}] alt={alt:.0f} km, "
                f"A={area:.4f} m²",
                end="",
                flush=True,
            )

            sample = {
                "N2_m3": n2,
                "O2_m3": o2,
                "O_m3": o_at,
                "N_m3": n_at,
                "T_K": T_K,
                "intake_area_m2": float(area),
                "altitude_km": float(alt),
                "orbital_speed_m_s": float(u_orb),
                "eta_collection": float(ETA_COLLECTION),
                "f107": float(f107),
                "f107a": float(f107a),
                "ap": float(ap),
                "lat_deg": float(lat),
                "lon_deg": float(lon),
                "date": date_ref.isoformat(),
            }

            try:
                thrust = float(_thrust_for_sample(sample, fast_mode=fast_mode))
                drag = drag_total_fmf(
                    rho=rho,
                    speed_m_s=u_orb,
                    intake_area_m2=area,
                    cd_body=CD_BODY,
                    a_body_m2=A_BODY_M2,
                )

                # Débit massique capté
                capture = ETA_COLLECTION * u_orb * area
                mdot = capture * rho
                isp = thrust / (mdot * 9.81) if mdot > 0 else 0.0

                thrust_map[i, j] = thrust
                drag_map[i, j] = drag
                margin_map[i, j] = thrust - drag
                isp_map[i, j] = isp
                mdot_map[i, j] = mdot

                print(f" → T={thrust:.2e} N, D={drag:.2e} N, Isp={isp:.0f} s")

            except Exception as exc:
                print(f" → FAILED: {exc}")
                thrust_map[i, j] = 0.0
                drag_map[i, j] = drag_total_fmf(
                    rho=rho,
                    speed_m_s=u_orb,
                    intake_area_m2=area,
                    cd_body=CD_BODY,
                    a_body_m2=A_BODY_M2,
                )
                margin_map[i, j] = -drag_map[i, j]

    return {
        "altitude_km": altitude_range_km,
        "area_m2": area_range_m2,
        "thrust_N": thrust_map,
        "drag_N": drag_map,
        "margin_N": margin_map,
        "isp_s": isp_map,
        "mdot_kg_s": mdot_map,
    }


def plot_ignition_map(results: dict, save_path: Path = None) -> None:
    alt = results["altitude_km"]
    area = results["area_m2"]
    thrust = results["thrust_N"]
    drag = results["drag_N"]
    margin = results["margin_N"]
    isp = results["isp_s"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ABEP Ignition & Performance Map", fontsize=14, fontweight="bold")

    area_grid, alt_grid = np.meshgrid(area, alt)

    # ── 1. Thrust map (log scale) ──
    ax = axes[0, 0]
    thrust_plot = np.where(thrust > 0, thrust, 1e-12)
    pcm = ax.pcolormesh(
        area_grid * 1e4,  # cm²
        alt_grid,
        thrust_plot,
        norm=mcolors.LogNorm(vmin=max(1e-8, np.nanmin(thrust_plot[thrust_plot > 0])),
                              vmax=np.nanmax(thrust_plot)),
        cmap="viridis",
        shading="nearest",
    )
    fig.colorbar(pcm, ax=ax, label="Thrust [N]")
    ax.set_xlabel("Intake area [cm²]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Thrust")
    ax.set_xscale("log")

    # ── 2. Margin map (thrust - drag) ──
    ax = axes[0, 1]
    margin_abs = np.abs(margin)
    margin_abs = np.where(margin_abs > 0, margin_abs, 1e-12)
    vmax = np.nanmax(margin_abs)

    pcm = ax.pcolormesh(
        area_grid * 1e4,
        alt_grid,
        margin,
        norm=mcolors.SymLogNorm(linthresh=1e-7, vmin=-vmax, vmax=vmax),
        cmap="RdYlGn",
        shading="nearest",
    )
    fig.colorbar(pcm, ax=ax, label="Thrust − Drag [N]")

    # Contour : margin = 0 (frontière de viabilité)
    try:
        ax.contour(
            area_grid * 1e4,
            alt_grid,
            margin,
            levels=[0.0],
            colors="black",
            linewidths=2,
            linestyles="--",
        )
    except Exception:
        pass

    ax.set_xlabel("Intake area [cm²]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Thrust − Drag (viability boundary = dashed)")
    ax.set_xscale("log")

    # ── 3. Isp map ──
    ax = axes[1, 0]
    isp_plot = np.where(isp > 10, isp, np.nan)  # masquer les points "éteints"
    pcm = ax.pcolormesh(
        area_grid * 1e4,
        alt_grid,
        isp_plot,
        cmap="plasma",
        shading="nearest",
    )
    fig.colorbar(pcm, ax=ax, label="Isp [s]")
    ax.set_xlabel("Intake area [cm²]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Isp (blanks = plasma off)")
    ax.set_xscale("log")

    # ── 4. Thrust / Drag ratio ──
    ax = axes[1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = thrust / drag
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    pcm = ax.pcolormesh(
        area_grid * 1e4,
        alt_grid,
        ratio,
        norm=mcolors.LogNorm(
            vmin=max(1e-3, np.nanmin(ratio[ratio > 0])),
            vmax=max(10, np.nanmax(ratio[np.isfinite(ratio)])),
        ),
        cmap="coolwarm",
        shading="nearest",
    )
    fig.colorbar(pcm, ax=ax, label="T/D ratio")

    # Contour T/D = 1
    try:
        ax.contour(
            area_grid * 1e4,
            alt_grid,
            ratio,
            levels=[1.0],
            colors="black",
            linewidths=2,
        )
    except Exception:
        pass

    ax.set_xlabel("Intake area [cm²]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("T/D ratio (contour = 1)")
    ax.set_xscale("log")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Scan and plot ABEP ignition map.")
    parser.add_argument("--alt-min", type=float, default=150.0)
    parser.add_argument("--alt-max", type=float, default=250.0)
    parser.add_argument("--alt-step", type=float, default=10.0)
    parser.add_argument("--area-min", type=float, default=0.01, help="Min intake area [m²]")
    parser.add_argument("--area-max", type=float, default=1.0, help="Max intake area [m²]")
    parser.add_argument("--area-points", type=int, default=8)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save figure.",
    )
    args = parser.parse_args()

    alt_range = np.arange(args.alt_min, args.alt_max + 0.1, args.alt_step)
    area_range = np.logspace(np.log10(args.area_min), np.log10(args.area_max), args.area_points)

    print(f"Scanning {len(alt_range)} altitudes × {len(area_range)} areas "
          f"= {len(alt_range) * len(area_range)} points")
    print(f"Altitudes: {alt_range}")
    print(f"Areas: {area_range}")

    results = scan_ignition_map(
        altitude_range_km=alt_range,
        area_range_m2=area_range,
        fast_mode=args.fast,
    )

    # Sauvegarder les données brutes
    scan_path = PROJECT_ROOT / "outputs" / "thrust_dataset" / "ignition_scan.json"
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    with scan_path.open("w") as f:
        json.dump(
            {
                "altitude_km": alt_range.tolist(),
                "area_m2": area_range.tolist(),
                "thrust_N": results["thrust_N"].tolist(),
                "drag_N": results["drag_N"].tolist(),
                "margin_N": results["margin_N"].tolist(),
                "isp_s": results["isp_s"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Scan data saved: {scan_path}")

    save_path = Path(args.save) if args.save else (
        PROJECT_ROOT / "outputs" / "thrust_dataset" / "ignition_map.png"
    )
    plot_ignition_map(results, save_path=save_path)


if __name__ == "__main__":
    main()
