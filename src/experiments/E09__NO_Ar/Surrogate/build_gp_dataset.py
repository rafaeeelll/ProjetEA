from __future__ import annotations

from surrogate_config import BASE_DATASET_PATH, GP_DATASET_PATH
from gp_utils import (
    make_feature_row,
    write_json,
    read_json,
    rho_from_species_densities,
    orbital_speed_from_altitude_km,
)


DEFAULT_INTAKE_AREA_M2 = 0.1


def main() -> None:
    data = read_json(BASE_DATASET_PATH)
    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"Dataset empty: {BASE_DATASET_PATH}")

    out_samples = []
    for s in samples:
        area = float(s.get("intake_area_m2", DEFAULT_INTAKE_AREA_M2))
        altitude_km = float(s.get("altitude_km", -1.0))
        if altitude_km <= 0.0:
            continue
        n2 = float(s["N2_m3"])
        o2 = float(s["O2_m3"])
        o = float(s["O_m3"])
        n = float(s["N_m3"])
        rho = rho_from_species_densities(n2, o2, o, n)
        speed = orbital_speed_from_altitude_km(altitude_km)
        x = make_feature_row(
            n2_m3=n2,
            o2_m3=o2,
            o_m3=o,
            n_m3=n,
            t_k=float(s["T_K"]),
            argon_injection_rate=float(s.get("argon_injection_rate", 0.0)),
            intake_area_m2=area,
            rho_kg_m3=rho,
            speed_m_s=speed,
        )
        out_samples.append(
            {
                "x": x,
                "y_total_thrust_N": float(s["total_thrust_N"]),
                "meta": {
                    "altitude_km": altitude_km,
                    "lat_deg": float(s.get("lat_deg", 0.0)),
                    "lon_deg": float(s.get("lon_deg", 0.0)),
                    "argon_injection_rate": float(s.get("argon_injection_rate", 0.0)),
                    "intake_area_m2": area,
                    "rho_kg_m3": rho,
                    "speed_m_s": speed,
                    "source": "thrust_dataset_msis",
                },
            }
        )

    payload = {
        "metadata": {
            "source_dataset": str(BASE_DATASET_PATH),
            "description": "GP thrust-only dataset with intake mass-flow feature (rho*u*A*eta_c).",
        },
        "samples": out_samples,
    }
    write_json(GP_DATASET_PATH, payload)
    print(f"Wrote {len(out_samples)} GP samples to {GP_DATASET_PATH}")


if __name__ == "__main__":
    main()
