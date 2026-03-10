#!/usr/bin/env python3
"""Trace l'évolution des densités atmosphériques en fonction de l'altitude à date/position fixées.

Même logique que le script temporel:
- courbes N, N2, O, O2, Total
- stats par espèce: minimum, moyenne, écart min-moyenne
- export figure + JSON

Exemple:
    python species_density_vs_altitude.py
    python species_density_vs_altitude.py --date 2020-01-01T12:00:00 --lat 0 --lon 0 --alt-min 100 --alt-max 500 --step-km 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


E09_DIR = Path(__file__).resolve().parents[1]
if str(E09_DIR) not in sys.path:
    sys.path.append(str(E09_DIR))

from msis_densities import (  # noqa: E402
    SPACE_WEATHER_PATH,
    _compute_f107a,
    _parse_space_weather,
    _space_weather_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évolution des densités N, N2, O, O2 en fonction de l'altitude à date/position fixées"
    )
    parser.add_argument("--date", type=str, default="2020-01-01T12:00:00", help="Date ISO (défaut: 2020-01-01T12:00:00)")
    parser.add_argument("--lat", type=float, default=0.0, help="Latitude fixe en degrés")
    parser.add_argument("--lon", type=float, default=0.0, help="Longitude fixe en degrés")
    parser.add_argument("--alt-min", type=float, default=100.0, help="Altitude min en km")
    parser.add_argument("--alt-max", type=float, default=500.0, help="Altitude max en km")
    parser.add_argument("--step-km", type=float, default=2.0, help="Pas altitude en km")
    parser.add_argument("--f107a", type=float, default=None, help="Override F10.7A (sinon space_weather)")
    parser.add_argument("--f107", type=float, default=None, help="Override F10.7 (sinon space_weather)")
    parser.add_argument("--ap", type=float, default=None, help="Override Ap (sinon space_weather)")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("figures", "E09", "atmosphere", "Fig_2_2_species_density_vs_altitude_fixed_point.png"),
        help="Chemin de sortie (png/pdf/svg)",
    )
    return parser.parse_args()


def build_altitude_axis(alt_min: float, alt_max: float, step_km: float) -> np.ndarray:
    if step_km <= 0:
        raise ValueError("--step-km doit être > 0")
    if alt_max <= alt_min:
        raise ValueError("--alt-max doit être strictement > --alt-min")

    alts = np.arange(alt_min, alt_max + 0.5 * step_km, step_km, dtype=float)
    if alts[-1] < alt_max:
        alts = np.append(alts, alt_max)
    return alts


def resolve_solar_params(
    dt: datetime,
    f107: float | None,
    f107a: float | None,
    ap: float | None,
    records: dict[date_cls, dict[str, object]],
    f107a_map: dict[date_cls, float],
) -> tuple[float, float, float]:
    if f107 is not None and f107a is not None and ap is not None:
        return float(f107), float(f107a), float(ap)
    if records:
        return _space_weather_params(dt, records, f107a_map)
    return 150.0, 150.0, 4.0


def query_species_density_m3(
    dt: datetime,
    alt_km: float,
    lat_deg: float,
    lon_deg: float,
    f107a: float,
    f107: float,
    ap: float,
) -> dict[str, float]:
    try:
        from nrlmsise00 import msise_model  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Impossible d'importer nrlmsise00. Installez avec: pip install nrlmsise00") from exc

    dens, _ = msise_model(dt, alt_km, lat_deg, lon_deg, f107a, f107, ap)
    dens = np.asarray(dens, dtype=float)

    species = {
        "O": dens[1] * 1e6,
        "N2": dens[2] * 1e6,
        "O2": dens[3] * 1e6,
        "N": dens[7] * 1e6,
    }
    species["Total"] = species["O"] + species["N2"] + species["O2"] + species["N"]
    return {k: float(v) for k, v in species.items()}


def compute_series_stats(altitudes_km: np.ndarray, series: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for species, values in series.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        min_idx = int(np.nanargmin(arr))
        min_val = float(arr[min_idx])
        mean_val = float(np.nanmean(arr))
        delta_abs = float(min_val - mean_val)
        delta_pct = float(100.0 * delta_abs / mean_val) if mean_val != 0 else float("nan")
        stats[species] = {
            "min_value_m3": min_val,
            "min_altitude_km": float(altitudes_km[min_idx]),
            "mean_value_m3": mean_val,
            "delta_min_minus_mean_m3": delta_abs,
            "delta_min_minus_mean_percent": delta_pct,
        }
    return stats


def main() -> None:
    args = parse_args()

    try:
        dt = datetime.fromisoformat(args.date)
    except ValueError:
        print("Erreur: format --date invalide. Utilisez par exemple 2020-01-01T12:00:00", file=sys.stderr)
        raise

    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)
    f107_res, f107a_res, ap_res = resolve_solar_params(
        dt=dt,
        f107=args.f107,
        f107a=args.f107a,
        ap=args.ap,
        records=records,
        f107a_map=f107a_map,
    )

    altitudes = build_altitude_axis(args.alt_min, args.alt_max, args.step_km)

    series = {"N": [], "N2": [], "O": [], "O2": [], "Total": []}
    for alt_km in altitudes:
        dens = query_species_density_m3(
            dt=dt,
            alt_km=float(alt_km),
            lat_deg=args.lat,
            lon_deg=args.lon,
            f107a=f107a_res,
            f107=f107_res,
            ap=ap_res,
        )
        for key in series:
            series[key].append(dens[key])

    plt.figure(figsize=(11, 6))
    plt.semilogy(altitudes, series["N2"], label="N2", linewidth=1.8)
    plt.semilogy(altitudes, series["O2"], label="O2", linewidth=1.8)
    plt.semilogy(altitudes, series["O"], label="O", linewidth=1.8)
    plt.semilogy(altitudes, series["N"], label="N", linewidth=1.8)

    plt.xlabel("Altitude (km)")
    plt.ylabel("Densité numérique (m⁻³)")
    plt.title(
        f"Densités vs altitude à date/position fixées | date={dt.isoformat(timespec='seconds')}\n"
        f"lat={args.lat:.1f}°, lon={args.lon:.1f}° | F10.7={f107_res:.1f}, F10.7A={f107a_res:.1f}, Ap={ap_res:.1f}"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    stats = compute_series_stats(altitudes, series)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

    total_arr = np.maximum(np.asarray(series["Total"], dtype=float), 1e-30)
    frac_fig_path = out_path.with_name("Fig_2_5_species_molar_fractions_vs_altitude_fixed_point.png")
    plt.figure(figsize=(11, 6))
    for species in ["N2", "O2", "O", "N"]:
        plt.plot(altitudes, np.asarray(series[species], dtype=float) / total_arr, label=species, linewidth=1.8)
    plt.xlabel("Altitude (km)")
    plt.ylabel("Fraction molaire [-]")
    plt.ylim(0.0, 1.0)
    plt.title(
        f"Fractions molaires vs altitude à date/position fixées | date={dt.isoformat(timespec='seconds')}\n"
        f"lat={args.lat:.1f}°, lon={args.lon:.1f}° | F10.7={f107_res:.1f}, F10.7A={f107a_res:.1f}, Ap={ap_res:.1f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(frac_fig_path, dpi=220)
    plt.close()

    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "date": dt.isoformat(timespec="seconds"),
                "lat_deg": float(args.lat),
                "lon_deg": float(args.lon),
                "alt_min_km": float(args.alt_min),
                "alt_max_km": float(args.alt_max),
                "step_km": float(args.step_km),
                "f107": float(f107_res),
                "f107a": float(f107a_res),
                "ap": float(ap_res),
                "stats": stats,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Figure sauvegardée: {out_path}")
    print(f"Figure fractions sauvegardée: {frac_fig_path}")
    print(f"Statistiques sauvegardées: {stats_path}")
    print("\nRésumé stats (min altitude, moyenne, écart min-moyenne):")
    for species in ["N2", "O2", "O", "N", "Total"]:
        if species not in stats:
            continue
        s = stats[species]
        print(
            f"- {species:5s} | min={float(s['min_value_m3']):.3e} à alt={float(s['min_altitude_km']):.1f} km | "
            f"moyenne={float(s['mean_value_m3']):.3e} | Δ={float(s['delta_min_minus_mean_m3']):.3e} "
            f"({float(s['delta_min_minus_mean_percent']):.2f}%)"
        )


if __name__ == "__main__":
    main()
