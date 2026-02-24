import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, k as k_B, pi
from scipy.interpolate import RBFInterpolator


MODEL_PATH = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent.joinpath("outputs", "thrust_dataset", "thrust_rbf_model.npz")
)
MODEL_META = MODEL_PATH.with_suffix(".json")


try:
    import global_model_package  # noqa: F401
except ModuleNotFoundError:
    global_model_package_path = Path(__file__).resolve().parent.parent.parent.joinpath("global_model_package")
    sys.path.append(str(global_model_package_path))

from global_model_package.chamber_caracteristics import Chamber
from global_model_package.model import GlobalModel
from global_model_package.reactions import ElectronHeatingConstantRFPower

from reaction_set_N_et_O import get_neutral_atmosphere, get_species_and_reactions


def _safe_log10(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _load_model() -> tuple[RBFInterpolator, np.ndarray, np.ndarray]:
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
    return rbf, data["x_mean"], data["x_std"]


def _load_msis_meta() -> dict:
    if not MODEL_META.exists():
        return {}
    with MODEL_META.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _msis_features(
    altitude_km: float,
    lat_deg: float,
    lon_deg: float,
    date: datetime,
    f107: float,
    f107a: float,
    ap: float,
) -> tuple[float, float, float, float, float]:
    from nrlmsise00 import msise_model  # type: ignore

    dens, temp = msise_model(date, altitude_km, lat_deg, lon_deg, f107a, f107, ap)
    dens = np.array(dens, dtype=float)
    temp = np.array(temp, dtype=float)

    n_N2 = float(dens[2]) * 1e6
    n_O2 = float(dens[3]) * 1e6
    n_O = float(dens[1]) * 1e6
    n_N = float(dens[7]) * 1e6
    T_K = float(temp[1])
    return n_N2, n_O2, n_O, n_N, T_K


def _sample_orbit(
    altitude_km: float,
    orbit_points: int,
    inclination_deg: float,
    raan_deg: float,
    lat0_deg: float,
    lon0_deg: float,
    date: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_earth = 3.986004418e14  # m^3/s^2
    r_earth = 6371e3  # m
    omega_earth = 7.2921159e-5  # rad/s

    radius = r_earth + altitude_km * 1000.0
    mean_motion = np.sqrt(mu_earth / radius**3)

    angles = np.linspace(0.0, 2.0 * np.pi, orbit_points)
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)

    if abs(np.sin(inc)) > 1e-8:
        sin_u0 = np.sin(np.radians(lat0_deg)) / np.sin(inc)
        sin_u0 = float(np.clip(sin_u0, -1.0, 1.0))
        u0 = float(np.arcsin(sin_u0))
    else:
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


def _predict_orbit(
    altitude_km: float,
    argon_injection_rate: float,
    orbit_points: int,
    inclination_deg: float,
    raan_deg: float,
    lat0_deg: float,
    lon0_deg: float,
    freeze_time: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = _load_msis_meta()
    msis = meta.get("msis", {})
    date = datetime.fromisoformat(msis.get("date", "2020-01-01T12:00:00"))
    f107 = float(msis.get("f107", 150.0))
    f107a = float(msis.get("f107a", 150.0))
    ap = float(msis.get("ap", 4.0))

    rbf, x_mean, x_std = _load_model()
    angles, lat_deg, lon_deg, time_s = _sample_orbit(
        altitude_km,
        orbit_points,
        inclination_deg,
        raan_deg,
        lat0_deg,
        lon0_deg,
        date,
    )

    feats = []
    for lat, lon, t_s in zip(lat_deg, lon_deg, time_s, strict=False):
        dt = date if freeze_time else date + timedelta(seconds=float(t_s))
        n2, o2, o, n, t_k = _msis_features(altitude_km, lat, lon, dt, f107, f107a, ap)
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
    X = np.array(feats, dtype=float)
    Xs = (X - x_mean) / x_std
    pred = rbf(Xs)
    return angles, lat_deg, lon_deg, time_s, pred


def _true_orbit(
    angles: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    time_s: np.ndarray,
    altitude_km: float,
    argon_injection_rate: float,
    freeze_time: bool,
) -> np.ndarray:
    meta = _load_msis_meta()
    msis = meta.get("msis", {})
    date = datetime.fromisoformat(msis.get("date", "2020-01-01T12:00:00"))
    f107 = float(msis.get("f107", 150.0))
    f107a = float(msis.get("f107a", 150.0))
    ap = float(msis.get("ap", 4.0))

    config_dict = {
        "R": 6e-2,
        "L": 10e-2,
        "V_grid": 1000,
        "beta_i": 0.7,
        "beta_g": 0.3,
        "omega": 13.56e6 * 2 * pi,
        "N": 5,
        "R_coil": 2,
    }

    chamber = Chamber(config_dict)
    thrust_list = []
    for theta, lat, lon, t_s in zip(angles, lat_deg, lon_deg, time_s, strict=False):
        dt = date if freeze_time else date + timedelta(seconds=float(t_s))
        atm = get_neutral_atmosphere(
            altitude_km,
            lat=lat,
            lon=lon,
            date=dt,
            f107=f107,
            f107a=f107a,
            ap=ap,
        )
        species, initial_state, reactions_list, _ = get_species_and_reactions(
            chamber,
            altitude_km,
            lat=lat,
            lon=lon,
            date=dt,
            ion_seed=1e10,
            electron_seed=1e12,
            compression_rate=500,
            collection_rate=0.5,
            argon_injection_rate=argon_injection_rate,
            atm=atm,
        )
        electron_heating = ElectronHeatingConstantRFPower(species, 1000, chamber)
        model = GlobalModel(
            species,
            reactions_list,
            chamber,
            electron_heating,
            simulation_name=f"rbf_true_theta_{theta:.3f}",
            log_folder_path=MODEL_PATH.parent,
            fast=True,
        )
        sol = model.solve(0.0, 1e-2, initial_state)
        final_state = sol.y[:, -1]
        thrust_list.append(float(model.total_thrust(final_state)))
    return np.array(thrust_list)


def plot_orbit_model_vs_true(
    altitude_km: float,
    argon_injection_rate: float,
    orbit_points: int = 50,
    inclination_deg: float = 51.6,
    raan_deg: float = 0.0,
    lat0_deg: float = 0.0,
    lon0_deg: float = 0.0,
    freeze_time: bool = True,
) -> None:
    angles, lat_deg, lon_deg, time_s, pred = _predict_orbit(
        altitude_km,
        argon_injection_rate,
        orbit_points,
        inclination_deg,
        raan_deg,
        lat0_deg,
        lon0_deg,
        freeze_time,
    )
    true_vals = _true_orbit(
        angles,
        lat_deg,
        lon_deg,
        time_s,
        altitude_km,
        argon_injection_rate,
        freeze_time,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(angles, pred, lw=2.0, label="RBF model")
    plt.plot(angles, true_vals, lw=2.0, linestyle="--", label="True model")
    plt.xlabel("Angle orbital (rad)")
    plt.ylabel("Thrust (N)")
    plt.title("Thrust vs Angle (RBF vs True)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_orbit_model_vs_true(
        altitude_km=190.0,
        argon_injection_rate=1e17,
        orbit_points=50,
        inclination_deg=51.6,
        raan_deg=0.0,
        lat0_deg=0.0,
        lon0_deg=0.0,
        freeze_time=True,
    )
