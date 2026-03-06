#!/usr/bin/env python3
"""Trace l'évolution temporelle des densités atmosphériques à position fixée.

Basé sur nrlmsise00 + paramètres space weather du projet.
Le script calcule N, N2, O, O2 (et la somme) au cours du temps pour une
latitude/longitude/altitude fixes.

Par défaut, il couvre toute la plage disponible de data/space_weather.txt.

Exemples:
    python species_density_vs_time.py
    python species_density_vs_time.py --no-full-range --start 2000-01-01T00:00:00 --years 10 --step-days 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date as date_cls
from datetime import datetime, timedelta
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
        description="Évolution temporelle des densités N, N2, O, O2 à position fixée"
    )
    parser.add_argument(
        "--full-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Utiliser toute la plage temporelle disponible dans space_weather.txt (défaut: true)",
    )
    parser.add_argument("--start", type=str, default="1950-01-01T00:00:00", help="Date de début ISO (utilisé si --no-full-range)")
    parser.add_argument("--years", type=float, default=10.0, help="Durée totale en années (utilisé si --no-full-range)")
    parser.add_argument("--step-days", type=float, default=30.0, help="Pas temporel en jours (défaut: 30)")
    parser.add_argument("--lat", type=float, default=0.0, help="Latitude fixe en degrés")
    parser.add_argument("--lon", type=float, default=0.0, help="Longitude fixe en degrés")
    parser.add_argument("--altitude", type=float, default=200.0, help="Altitude fixe en km")
    parser.add_argument("--f107a", type=float, default=None, help="Override F10.7A (sinon space_weather)")
    parser.add_argument("--f107", type=float, default=None, help="Override F10.7 (sinon space_weather)")
    parser.add_argument("--ap", type=float, default=None, help="Override Ap (sinon space_weather)")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("figures", "E09", "atmosphere", "species_density_vs_time_fixed_point.png"),
        help="Chemin de sortie (png/pdf/svg)",
    )
    return parser.parse_args()


def _available_date_range() -> tuple[date_cls, date_cls]:
    records = _parse_space_weather(SPACE_WEATHER_PATH)
    if not records:
        raise ValueError(f"Aucune donnée disponible dans {SPACE_WEATHER_PATH}")
    dates = sorted(records.keys())
    return dates[0], dates[-1]


def build_time_axis(start: datetime, end: datetime, step_days: float) -> list[datetime]:
    if step_days <= 0:
        raise ValueError("--step-days doit être > 0")
    if end <= start:
        raise ValueError("La date de fin doit être strictement après la date de début")

    step = timedelta(days=step_days)
    timeline = []
    current = start
    while current <= end:
        timeline.append(current)
        current += step
    if timeline[-1] < end:
        timeline.append(end)
    return timeline


def query_species_density_m3(
    dt: datetime,
    alt_km: float,
    lat_deg: float,
    lon_deg: float,
    f107a: float | None,
    f107: float | None,
    ap: float | None,
    records: dict[date_cls, dict[str, object]] | None = None,
    f107a_map: dict[date_cls, float] | None = None,
) -> dict[str, float]:
    try:
        from nrlmsise00 import msise_model  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Impossible d'importer nrlmsise00. Installez avec: pip install nrlmsise00") from exc

    if f107 is None or f107a is None or ap is None:
        recs = records if records is not None else _parse_space_weather(SPACE_WEATHER_PATH)
        f107a_vals = f107a_map if f107a_map is not None else _compute_f107a(recs)
        if recs:
            f107_res, f107a_res, ap_res = _space_weather_params(dt, recs, f107a_vals)
        else:
            f107_res, f107a_res, ap_res = 150.0, 150.0, 4.0
    else:
        f107_res, f107a_res, ap_res = float(f107), float(f107a), float(ap)

    dens, _ = msise_model(dt, alt_km, lat_deg, lon_deg, f107a_res, f107_res, ap_res)
    dens = np.asarray(dens, dtype=float)

    # nrlmsise00: dens en cm^-3 pour les espèces neutres
    species = {
        "O": dens[1] * 1e6,
        "N2": dens[2] * 1e6,
        "O2": dens[3] * 1e6,
        "N": dens[7] * 1e6,
    }
    species["Total"] = species["O"] + species["N2"] + species["O2"] + species["N"]
    return {k: float(v) for k, v in species.items()}


def compute_series_stats(timeline: list[datetime], series: dict[str, list[float]]) -> dict[str, dict[str, float | str]]:
    stats: dict[str, dict[str, float | str]] = {}
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
            "min_time_iso": timeline[min_idx].isoformat(timespec="seconds"),
            "mean_value_m3": mean_val,
            "delta_min_minus_mean_m3": delta_abs,
            "delta_min_minus_mean_percent": delta_pct,
        }
    return stats


def _interp_fill(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    n = out.size
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    valid_idx = np.where(valid)[0]
    for i in range(n):
        if valid[i]:
            continue
        left = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]
        if left.size > 0 and right.size > 0:
            li = int(left[-1])
            ri = int(right[0])
            out[i] = float(np.exp(0.5 * (np.log(out[li]) + np.log(out[ri]))))
        elif left.size > 0:
            out[i] = float(out[int(left[-1])])
        elif right.size > 0:
            out[i] = float(out[int(right[0])])
    return out


def despike_series(
    timeline: list[datetime],
    series: dict[str, list[float]],
    low_ratio: float = 1e-6,
    high_ratio: float = 1e6,
    half_window: int = 3,
) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    cleaned: dict[str, list[float]] = {}
    flagged: dict[str, list[str]] = {}

    for species, values in series.items():
        arr = np.asarray(values, dtype=float)
        n = arr.size
        bad = np.zeros(n, dtype=bool)

        # Invalid values are always flagged
        bad |= ~np.isfinite(arr)
        bad |= arr <= 0.0

        # Local spike detection against neighboring median
        for i in range(n):
            lo = max(0, i - half_window)
            hi = min(n, i + half_window + 1)
            neighborhood = arr[lo:hi]
            if neighborhood.size < 3:
                continue
            neighbors = np.delete(neighborhood, min(i - lo, neighborhood.size - 1))
            neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
            if neighbors.size < 2:
                continue
            local_med = float(np.median(neighbors))
            if local_med <= 0:
                continue
            if arr[i] < local_med * low_ratio or arr[i] > local_med * high_ratio:
                bad[i] = True

        arr_clean = arr.copy()
        arr_clean[bad] = np.nan
        arr_clean = _interp_fill(arr_clean)
        cleaned[species] = arr_clean.tolist()
        flagged[species] = [timeline[i].isoformat(timespec="seconds") for i in np.where(bad)[0]]

    return cleaned, flagged


def main() -> None:
    args = parse_args()
    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)

    if args.full_range:
        dmin, dmax = _available_date_range()
        start_floor = date_cls(1950, 1, 1)
        effective_start = dmin if dmin >= start_floor else start_floor
        start = datetime(effective_start.year, effective_start.month, effective_start.day, 0, 0, 0)
        end = datetime(dmax.year, dmax.month, dmax.day, 0, 0, 0)
    else:
        try:
            start = datetime.fromisoformat(args.start)
        except ValueError:
            print("Erreur: format --start invalide. Utilisez par exemple 2020-01-01T00:00:00", file=sys.stderr)
            raise
        end = start + timedelta(days=args.years * 365.25)

    timeline = build_time_axis(start, end=end, step_days=args.step_days)

    series = {"N": [], "N2": [], "O": [], "O2": [], "Total": []}

    for dt in timeline:
        dens = query_species_density_m3(
            dt=dt,
            alt_km=args.altitude,
            lat_deg=args.lat,
            lon_deg=args.lon,
            f107a=args.f107a,
            f107=args.f107,
            ap=args.ap,
            records=records,
            f107a_map=f107a_map,
        )
        for key in series:
            series[key].append(dens[key])

    series_clean, flagged = despike_series(timeline, series)

    plt.figure(figsize=(11, 6))
    plt.semilogy(timeline, series_clean["N2"], label="N2", linewidth=1.8)
    plt.semilogy(timeline, series_clean["O2"], label="O2", linewidth=1.8)
    plt.semilogy(timeline, series_clean["O"], label="O", linewidth=1.8)
    plt.semilogy(timeline, series_clean["N"], label="N", linewidth=1.8)
    plt.semilogy(timeline, series_clean["Total"], label="Total", linestyle="--", linewidth=2.0, color="black")

    plt.xlabel("Date")
    plt.ylabel("Densité numérique (m⁻³)")
    plt.title(
        f"Évolution des densités à position fixe | lat={args.lat:.1f}°, lon={args.lon:.1f}°, alt={args.altitude:.1f} km\n"
        f"Plage: {start.date().isoformat()} → {end.date().isoformat()} | pas={args.step_days:g} jours"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    stats = compute_series_stats(timeline, series_clean)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "start": start.isoformat(timespec="seconds"),
                "end": end.isoformat(timespec="seconds"),
                "step_days": float(args.step_days),
                "lat_deg": float(args.lat),
                "lon_deg": float(args.lon),
                "altitude_km": float(args.altitude),
                "despike": {
                    "method": "local_median_ratio",
                    "low_ratio": 1e-6,
                    "high_ratio": 1e6,
                    "half_window": 3,
                    "flagged_timestamps_by_species": flagged,
                },
                "stats": stats,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Figure sauvegardée: {out_path}")
    print(f"Statistiques sauvegardées: {stats_path}")
    print("\nRésumé stats (min, moyenne, écart min-moyenne):")
    for species in ["N2", "O2", "O", "N", "Total"]:
        if species not in stats:
            continue
        s = stats[species]
        print(
            f"- {species:5s} | min={float(s['min_value_m3']):.3e} à {s['min_time_iso']} | "
            f"moyenne={float(s['mean_value_m3']):.3e} | Δ={float(s['delta_min_minus_mean_m3']):.3e} "
            f"({float(s['delta_min_minus_mean_percent']):.2f}%)"
        )


if __name__ == "__main__":
    main()
