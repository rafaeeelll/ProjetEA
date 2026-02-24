import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator


MODEL_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_rbf_model.npz")
)
MODEL_META = MODEL_PATH.with_suffix(".json")
OUTPUT_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "orbit_optimization.json")
)
DATASET_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")
)


# --- Optimization grid (full-domain defaults) ---
USE_DATASET_ALTITUDES = True
ALTITUDE_STEP_KM = 5.0
INCLINATION_STEP_DEG = 5.0
RAAN_STEP_DEG = 10.0

ORBIT_POINTS = 60
ARGON_INJECTION_RATE = 1e17
FREEZE_TIME = True

# --- Drag model (simple, tunable) ---
CHAMBER_RADIUS_M = 6e-2
CHAMBER_LENGTH_M = 10e-2
ARRAY_AREA_M2 = 1

CD_FRONT = 2.2
CD_SIDE = 2.2
CD_ARRAY = 2.2
ETA_C = 0.35
USE_INTAKE_MOMENTUM = False
MIN_MARGIN_N = 0.0


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _load_model() -> tuple[RBFInterpolator, np.ndarray, np.ndarray, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    data = np.load(MODEL_PATH)
    rbf = RBFInterpolator(
        data["Xs"],
        data["y"],
        kernel=str(data["kernel"]),
        smoothing=float(data["smoothing"]),
        neighbors=int(data["neighbors"]),
    )
    meta = {}
    if MODEL_META.exists():
        with MODEL_META.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    return rbf, data["x_mean"], data["x_std"], meta


def _load_altitudes() -> np.ndarray:
    if not DATASET_PATH.exists():
        if USE_DATASET_ALTITUDES:
            raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")
        return np.arange(170.0, 221.0, ALTITUDE_STEP_KM)
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    meta = data.get("metadata", {})
    alts = meta.get("altitudes_km")
    if USE_DATASET_ALTITUDES and alts:
        return np.array(alts, dtype=float)
    samples = data.get("samples", [])
    if not samples:
        return np.arange(170.0, 221.0, ALTITUDE_STEP_KM)
    unique_alts = sorted({float(s["altitude_km"]) for s in samples})
    return np.array(unique_alts, dtype=float)


def _orbit_samples(
    altitude_km: float,
    orbit_points: int,
    inclination_deg: float,
    raan_deg: float,
    date: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_earth = 3.986004418e14  # m^3/s^2
    r_earth = 6371e3  # m
    omega_earth = 7.2921159e-5  # rad/s

    radius = r_earth + altitude_km * 1000.0
    mean_motion = np.sqrt(mu_earth / radius**3)

    angles = np.linspace(0.0, 2.0 * np.pi, orbit_points, endpoint=False)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    u0 = 0.0

    lat_list = []
    lon_list = []
    time_list = []
    for theta in angles:
        u = u0 + theta
        t = u / mean_motion
        time_list.append(float(t))

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

        lat_list.append(float(np.degrees(lat_rad)))
        lon_list.append(float(np.degrees(lon_rad)))

    return angles, np.array(lat_list), np.array(lon_list), np.array(time_list)


def _msis_features(
    altitude_km: float,
    lat_deg: float,
    lon_deg: float,
    date: datetime,
    f107: float,
    f107a: float,
    ap: float,
) -> tuple[float, float, float, float, float, float]:
    from nrlmsise00 import msise_model  # type: ignore

    dens, temp = msise_model(date, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
    dens = np.array(dens, dtype=float)
    temp = np.array(temp, dtype=float)

    n_N2 = float(dens[2]) * 1e6
    n_O2 = float(dens[3]) * 1e6
    n_O = float(dens[1]) * 1e6
    n_N = float(dens[7]) * 1e6
    T_K = float(temp[1])
    # kg/m^3 using species number densities
    m_N2 = 4.65e-26
    m_O2 = 5.31e-26
    m_O = 2.67e-26
    m_N = 2.33e-26
    rho = n_N2 * m_N2 + n_O2 * m_O2 + n_O * m_O + n_N * m_N
    return n_N2, n_O2, n_O, n_N, T_K, rho


def _predict_orbit_thrust_series(
    rbf: RBFInterpolator,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    argon_injection_rate: float,
    date: datetime,
    f107: float,
    f107a: float,
    ap: float,
    orbit_points: int,
    freeze_time: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles, lat_deg, lon_deg, time_s = _orbit_samples(
        altitude_km, orbit_points, inclination_deg, raan_deg, date
    )

    feats = []
    rho_list = []
    for lat, lon, t_s in zip(lat_deg, lon_deg, time_s, strict=False):
        dt = date if freeze_time else date + timedelta(seconds=float(t_s))
        n2, o2, o, n, t_k, rho = _msis_features(
            altitude_km, lat, lon, dt, f107, f107a, ap
        )
        feats.append(
            [
                _safe_log10(np.array([n2]))[0],
                _safe_log10(np.array([o2]))[0],
                _safe_log10(np.array([o]))[0],
                _safe_log10(np.array([n]))[0],
                t_k,
                _safe_log10(np.array([argon_injection_rate]))[0],
            ]
        )
        rho_list.append(rho)
    X = np.array(feats, dtype=float)
    Xs = (X - x_mean) / x_std
    thrust = rbf(Xs)
    return thrust, np.array(rho_list), lat_deg, lon_deg


def _orbital_speed(altitude_km: float) -> float:
    mu_earth = 3.986004418e14  # m^3/s^2
    r_earth = 6371e3  # m
    radius = r_earth + altitude_km * 1000.0
    return float(np.sqrt(mu_earth / radius))


def _drag_force(rho: np.ndarray, u_inf: float) -> np.ndarray:
    a_front = np.pi * CHAMBER_RADIUS_M**2
    a_side = 2.0 * np.pi * CHAMBER_RADIUS_M * CHAMBER_LENGTH_M
    a_array = ARRAY_AREA_M2

    eff_front = (1.0 - ETA_C) * a_front
    cd_area = CD_FRONT * eff_front + CD_SIDE * a_side + CD_ARRAY * a_array
    drag = 0.5 * rho * u_inf**2 * cd_area

    if USE_INTAKE_MOMENTUM:
        mdot = ETA_C * rho * a_front * u_inf
        drag = drag + mdot * u_inf
    return drag


def main() -> None:
    rbf, x_mean, x_std, meta = _load_model()
    msis = meta.get("msis", {})
    date = datetime.fromisoformat(msis.get("date", "2020-01-01T12:00:00"))
    f107 = float(msis.get("f107", 150.0))
    f107a = float(msis.get("f107a", 150.0))
    ap = float(msis.get("ap", 4.0))

    altitudes_km = _load_altitudes()
    inclinations_deg = np.arange(0.0, 180.0 + 1e-9, INCLINATION_STEP_DEG)
    raan_deg = np.arange(0.0, 360.0, RAAN_STEP_DEG)

    results = []
    best = None

    total = len(altitudes_km) * len(inclinations_deg) * len(raan_deg)
    idx = 0
    for alt in altitudes_km:
        for inc in inclinations_deg:
            for raan in raan_deg:
                idx += 1
                thrust, rho, _, _ = _predict_orbit_thrust_series(
                    rbf,
                    x_mean,
                    x_std,
                    altitude_km=float(alt),
                    inclination_deg=float(inc),
                    raan_deg=float(raan),
                    argon_injection_rate=float(ARGON_INJECTION_RATE),
                    date=date,
                    f107=f107,
                    f107a=f107a,
                    ap=ap,
                    orbit_points=ORBIT_POINTS,
                    freeze_time=FREEZE_TIME,
                )
                u_inf = _orbital_speed(float(alt))
                drag = _drag_force(rho, u_inf)
                margin = thrust - drag
                mean_thrust = float(np.mean(thrust))
                mean_drag = float(np.mean(drag))
                min_margin = float(np.min(margin))
                feasible = min_margin >= MIN_MARGIN_N
                objective = float(np.mean(margin)) if feasible else float("-inf")
                row = {
                    "altitude_km": float(alt),
                    "inclination_deg": float(inc),
                    "raan_deg": float(raan),
                    "mean_thrust_N": mean_thrust,
                    "mean_drag_N": mean_drag,
                    "min_margin_N": min_margin,
                    "objective": objective,
                    "feasible": feasible,
                }
                results.append(row)
                if best is None or objective > best["objective"]:
                    best = row
                if idx % 20 == 0:
                    print(f"[{idx}/{total}] best so far: {best}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "argon_injection_rate": float(ARGON_INJECTION_RATE),
                "orbit_points": int(ORBIT_POINTS),
                "freeze_time": bool(FREEZE_TIME),
                "grid": {
                    "altitudes_km": altitudes_km.tolist(),
                    "inclinations_deg": inclinations_deg.tolist(),
                    "raan_deg": raan_deg.tolist(),
                },
                "best": best,
                "results": results,
            },
            handle,
            indent=2,
        )

    print("Best orbit:", best)


if __name__ == "__main__":
    main()
