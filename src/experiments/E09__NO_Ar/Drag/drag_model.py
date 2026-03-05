from __future__ import annotations

import numpy as np

# Species masses [kg]
M_N2 = 4.65e-26
M_O2 = 5.31e-26
M_O = 2.67e-26
M_N = 2.33e-26

# Shared drag defaults
ETA_COLLECTION_DEFAULT = 0.5
CD_BODY_DEFAULT = 2.2
A_BODY_M2_DEFAULT = 0.01
R_CHAMBER_DEFAULT_M = 6e-2
ETA_C_MAX = ETA_COLLECTION_DEFAULT
ETA_C_ALPHA = 0.3


def collection_efficiency(A_intake: float, A_chamber: float | None = None) -> float:
    """
    Intake collection efficiency model as a function of intake area.
    """
    if A_chamber is None:
        A_chamber = np.pi * R_CHAMBER_DEFAULT_M**2
    ar = max(float(A_intake) / float(A_chamber), 1.0)
    return float(ETA_C_MAX / (1.0 + ETA_C_ALPHA * np.log(ar)))


def mass_density_from_number_densities(
    n2_m3: float,
    o2_m3: float,
    o_m3: float,
    n_m3: float,
) -> float:
    """Mass density [kg/m^3] from species number densities [m^-3]."""
    return (
        float(n2_m3) * M_N2
        + float(o2_m3) * M_O2
        + float(o_m3) * M_O
        + float(n_m3) * M_N
    )


def mass_density_from_sample(sample: dict) -> float:
    """Mass density [kg/m^3] from a sample dict carrying N2/O2/O/N number densities."""
    return mass_density_from_number_densities(
        n2_m3=float(sample["N2_m3"]),
        o2_m3=float(sample["O2_m3"]),
        o_m3=float(sample["O_m3"]),
        n_m3=float(sample["N_m3"]),
    )


def drag_classic(rho: float, speed_m_s: float, cd: float, area_m2: float) -> float:
    """Classical aerodynamic convention: 0.5 * rho * u^2 * Cd * A."""
    return 0.5 * float(rho) * float(speed_m_s) ** 2 * float(cd) * float(area_m2)


def drag_intake_fmf(
    rho: float,
    speed_m_s: float,
    intake_area_m2: float,
    a_chamber_m2: float | None = None,
) -> float:
    """
    Free-molecular intake drag with captured vs reflected particle split.
    F_intake = rho * u^2 * A_intake * (2 - eta_c)
    """
    eta_collection = collection_efficiency(
        A_intake=float(intake_area_m2),
        A_chamber=a_chamber_m2,
    )
    return (
        float(rho)
        * float(speed_m_s) ** 2
        * float(intake_area_m2)
        * (2.0 - float(eta_collection))
    )


def drag_body_fmf(
    rho: float,
    speed_m_s: float,
    cd_body: float = CD_BODY_DEFAULT,
    a_body_m2: float = A_BODY_M2_DEFAULT,
) -> float:
    """Satellite body drag with classical convention."""
    return drag_classic(rho=float(rho), speed_m_s=float(speed_m_s), cd=float(cd_body), area_m2=float(a_body_m2))


def drag_total_fmf(
    rho: float,
    speed_m_s: float,
    intake_area_m2: float,
    cd_body: float = CD_BODY_DEFAULT,
    a_body_m2: float = A_BODY_M2_DEFAULT,
    a_chamber_m2: float | None = None,
) -> float:
    """Total drag = intake FMF term + body term."""
    return drag_intake_fmf(
        rho=float(rho),
        speed_m_s=float(speed_m_s),
        intake_area_m2=float(intake_area_m2),
        a_chamber_m2=a_chamber_m2,
    ) + drag_body_fmf(
        rho=float(rho),
        speed_m_s=float(speed_m_s),
        cd_body=float(cd_body),
        a_body_m2=float(a_body_m2),
    )


def drag_total_from_sample(
    sample: dict,
    intake_area_key: str = "intake_area_m2",
    speed_key: str = "orbital_speed_m_s",
    cd_body: float = CD_BODY_DEFAULT,
    a_body_m2: float = A_BODY_M2_DEFAULT,
    a_chamber_m2: float | None = None,
) -> float:
    """Total FMF drag from a sample dictionary."""
    rho = mass_density_from_sample(sample)
    speed_m_s = float(sample[speed_key])
    intake_area_m2 = float(sample[intake_area_key])
    return drag_total_fmf(
        rho=rho,
        speed_m_s=speed_m_s,
        intake_area_m2=intake_area_m2,
        cd_body=cd_body,
        a_body_m2=a_body_m2,
        a_chamber_m2=a_chamber_m2,
    )
