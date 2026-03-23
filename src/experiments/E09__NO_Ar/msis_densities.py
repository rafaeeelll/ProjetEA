"""Centralise la lecture du space weather et les appels MSIS pour E09.

Comme plusieurs scripts en dependent, on garde ici une base commune pour sortir
des densites coherentes selon la date, l'altitude et la position.
"""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.constants import e, k as k_B


SPACE_WEATHER_PATH = (
    Path(__file__).resolve().parents[3].joinpath("data", "space_weather.txt")
)
FIGURES_DIR = Path(__file__).resolve().parents[3].joinpath("figures", "E09")


def _parse_space_weather(path: Path) -> dict[date_cls, dict[str, object]]:
    records: dict[date_cls, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 28:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
            except ValueError:
                continue
            day_key = date_cls(year, month, day)
            try:
                ap_3h = [int(p) for p in parts[15:23]]
                ap_daily = float(parts[23])
                f107_obs = float(parts[25])
                f107_adj = float(parts[26])
            except ValueError:
                continue
            records[day_key] = {
                "ap_3h": ap_3h,
                "ap_daily": ap_daily,
                "f107_obs": f107_obs,
                "f107_adj": f107_adj,
            }
    return records


def _compute_f107a(records: dict[date_cls, dict[str, object]]) -> dict[date_cls, float]:
    if not records:
        return {}
    dates_sorted = sorted(records.keys())
    f107a_map: dict[date_cls, float] = {}
    for idx, day_key in enumerate(dates_sorted):
        window_start = max(0, idx - 80)
        window_dates = dates_sorted[window_start : idx + 1]
        vals = []
        for d in window_dates:
            v = float(records[d]["f107_obs"])
            if v >= 0:
                vals.append(v)
        if vals:
            f107a_map[day_key] = float(np.mean(vals))
        else:
            f107a_map[day_key] = float(records[day_key]["f107_adj"])
    return f107a_map


def _nearest_available_date(target: date_cls, records: dict[date_cls, dict[str, object]]) -> date_cls:
    if target in records:
        return target
    if not records:
        raise ValueError("space_weather.txt has no data rows.")
    dates_sorted = sorted(records.keys())
    if target < dates_sorted[0]:
        return dates_sorted[0]
    if target > dates_sorted[-1]:
        return dates_sorted[-1]
    # fallback to the closest previous date
    for d in reversed(dates_sorted):
        if d <= target:
            return d
    return dates_sorted[0]


def _space_weather_params(
    dt: datetime,
    records: dict[date_cls, dict[str, object]],
    f107a_map: dict[date_cls, float],
) -> tuple[float, float, float]:
    day_key = _nearest_available_date(dt.date(), records)
    rec = records[day_key]
    f107_obs = float(rec["f107_obs"])
    f107_adj = float(rec["f107_adj"])
    f107 = f107_obs if f107_obs >= 0 else f107_adj
    f107a = f107a_map.get(day_key, f107_adj)
    ap_daily = float(rec["ap_daily"])
    if ap_daily < 0:
        ap_3h = rec.get("ap_3h", [])
        ap_vals = [float(v) for v in ap_3h if float(v) >= 0]
        if ap_vals:
            ap_daily = float(np.mean(ap_vals))
    return f107, f107a, ap_daily


@lru_cache(maxsize=1)
def _load_space_weather() -> tuple[dict[date_cls, dict[str, object]], dict[date_cls, float]]:
    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)
    return records, f107a_map


def resolve_space_weather_params(
    dt: datetime,
    f107: float | None = None,
    f107a: float | None = None,
    ap: float | None = None,
) -> tuple[float, float, float]:
    if f107 is not None and f107a is not None and ap is not None:
        return float(f107), float(f107a), float(ap)
    records, f107a_map = _load_space_weather()
    if not records:
        return 150.0, 150.0, 4.0
    return _space_weather_params(dt, records, f107a_map)


def get_msis_neutral_atmosphere(
    altitude_km: float,
    lat: float = 0.0,
    lon: float = 0.0,
    date: datetime = datetime(2020, 1, 1, 12, 0, 0),
    f107: float | None = None,
    f107a: float | None = None,
    ap: float | None = None,
) -> dict[str, float]:
    from nrlmsise00 import msise_model  # type: ignore

    f107_res, f107a_res, ap_res = resolve_space_weather_params(date, f107=f107, f107a=f107a, ap=ap)
    dens, temp = msise_model(date, altitude_km, lat, lon, f107a_res, f107_res, ap_res)
    dens = np.array(dens, dtype=float)
    temp = np.array(temp, dtype=float)

    n_N2 = float(dens[2]) * 1e6
    n_O2 = float(dens[3]) * 1e6
    n_O = float(dens[1]) * 1e6
    n_N = float(dens[7]) * 1e6
    n_total = n_N2 + n_O2 + n_O + n_N

    t_k = float(temp[1])
    return {
        "N2": n_N2,
        "N": n_N,
        "O2": n_O2,
        "O": n_O,
        "T_K": t_k,
        "T_eV": float(k_B * t_k / e),
        "pressure_pa": float(n_total * k_B * t_k),
        "source": "msise",
        "f107": float(f107_res),
        "f107a": float(f107a_res),
        "ap": float(ap_res),
    }


def main():
    import matplotlib.pyplot as plt

    try:
        from nrlmsise00 import msise_model  # type: ignore
    except Exception as exc:
        print(f"Import error: {exc}")
        return

    altitude_km = 250.0
    lat = 0.0
    lon = 0.0
    inclination_deg = 51.6
    raan_deg = lon
    date = datetime(2020, 1, 1, 12, 0, 0)
    orbit_points = 180

    records = _parse_space_weather(SPACE_WEATHER_PATH)
    f107a_map = _compute_f107a(records)

    f107, f107a, ap = _space_weather_params(date, records, f107a_map)

    dens, temp = msise_model(date, altitude_km, lat, lon, f107a, f107, ap)

    print("MSIS output (raw):")
    print("densities:", dens)
    print("temperatures:", temp)

    # Typical indices used in many wrappers:
    # d[1]=O, d[2]=N2, d[3]=O2, d[7]=N (in cm^-3 in some implementations)
    try:
        print(f"O  (d[1]): {dens[1]} cm^-3")
        print(f"N2 (d[2]): {dens[2]} cm^-3")
        print(f"O2 (d[3]): {dens[3]} cm^-3")
        print(f"N  (d[7]): {dens[7]} cm^-3")
    except Exception:
        pass

    # ---- Orbite circulaire: densités par espèce vs angle ----
    # Hypothèses: orbite circulaire, vitesse constante, inclinaison fixe.
    # On tient compte de la rotation terrestre pour approximer le ground track.
    mu_earth = 3.986004418e14  # m^3/s^2
    r_earth = 6371e3  # m
    omega_earth = 7.2921159e-5  # rad/s

    radius = r_earth + altitude_km * 1000.0
    mean_motion = np.sqrt(mu_earth / radius**3)

    angles = np.linspace(0.0, 2.0 * np.pi, orbit_points)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)

    if abs(np.sin(inc)) > 1e-8:
        sin_u0 = np.sin(np.radians(lat)) / np.sin(inc)
        sin_u0 = float(np.clip(sin_u0, -1.0, 1.0))
        u0 = float(np.arcsin(sin_u0))
    else:
        u0 = 0.0

    density_O = []
    density_N = []
    density_N2 = []
    density_O2 = []
    density_Ar = []
    for theta in angles:
        u = u0 + theta
        t = u / mean_motion
        dt = date + timedelta(seconds=float(t))
        f107, f107a, ap = _space_weather_params(dt, records, f107a_map)

        x_orb = radius * np.cos(u)
        y_orb = radius * np.sin(u)

        x_inc = x_orb
        y_inc = y_orb * np.cos(inc)
        z_inc = y_orb * np.sin(inc)

        x_eci = x_inc * np.cos(raan) - y_inc * np.sin(raan)
        y_eci = x_inc * np.sin(raan) + y_inc * np.cos(raan)
        z_eci = z_inc

        cos_earth = np.cos(omega_earth * t)
        sin_earth = np.sin(omega_earth * t)
        x_ecef = x_eci * cos_earth + y_eci * sin_earth
        y_ecef = -x_eci * sin_earth + y_eci * cos_earth
        z_ecef = z_eci

        lat_rad = np.arcsin(z_ecef / radius)
        lon_rad = np.arctan2(y_ecef, x_ecef)

        lat_deg = float(np.degrees(lat_rad))
        lon_deg = float(np.degrees(lon_rad))

        dens_orb, _ = msise_model(dt, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
        dens_orb = np.array(dens_orb, dtype=float)

        n_O = float(dens_orb[1]) * 1.0e6
        n_N2 = float(dens_orb[2]) * 1.0e6
        n_O2 = float(dens_orb[3]) * 1.0e6
        n_Ar = float(dens_orb[4]) * 1.0e6
        n_N = float(dens_orb[7]) * 1.0e6
        if n_N <= 0.0 and dens_orb.size > 6:
            n_N = float(dens_orb[6]) * 1.0e6

        density_O.append(n_O)
        density_N.append(n_N)
        density_N2.append(n_N2)
        density_O2.append(n_O2)
        density_Ar.append(n_Ar)

    plt.figure(figsize=(8.0, 4.8))
    plt.plot(angles, density_O, lw=2.0, label="O")
    plt.plot(angles, density_N, lw=2.0, label="N")
    plt.plot(angles, density_N2, lw=2.0, label="N2")
    plt.plot(angles, density_O2, lw=2.0, label="O2")
    plt.plot(angles, density_Ar, lw=2.0, label="Ar")
    plt.xlabel("Angle orbital (rad)")
    plt.ylabel("Densité (m$^{-3}$)")
    plt.yscale("log")
    plt.title("Densité MSIS par espèce vs angle (orbite circulaire)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR.joinpath("Fig_2_3_species_density_vs_orbit_angle.png")
    plt.savefig(save_path, dpi=220)
    plt.show()
    print(f"Figure sauvegardee: {save_path}")


if __name__ == "__main__":
    main()
