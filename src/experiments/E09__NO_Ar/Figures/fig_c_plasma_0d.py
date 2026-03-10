from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, Case, build_sample_from_case, solve_plasma_sample, weather_records


def _species_idx(species, name: str) -> int | None:
    try:
        return int(species.get_specie_by_name(name).index)
    except Exception:
        return None


def figure_7_time_evolution(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    cases = [
        ("on", Case(altitude_km=180.0, area_m2=0.1)),
        ("off", Case(altitude_km=240.0, area_m2=0.01)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    for row, (label, case) in enumerate(cases):
        sample = build_sample_from_case(case, records=records, f107a_map=f107a_map)
        res = solve_plasma_sample(sample, power_rf_w=1000.0, fast_mode=fast_mode, simulation_name=f"fig7_{label}")
        species = res["species"]
        sol = res["solution"]
        states = res["states"]
        t = np.asarray(sol.t, dtype=float)
        te = np.maximum(states[:, species.nb], 1e-30)
        tmono = np.maximum(states[:, species.nb + 1], 1e-30)
        tdiato = np.maximum(states[:, species.nb + 2], 1e-30)
        ne = np.maximum(states[:, 0], 1e-30)
        n_total = np.maximum(np.sum(states[:, : species.nb], axis=1), 1e-30)
        eta_ion = np.maximum(ne / n_total, 1e-30)

        for name in ["e", "N2", "O", "N2+", "O+"]:
            idx = _species_idx(species, name)
            if idx is not None:
                axes[row, 0].plot(t, np.maximum(states[:, idx], 1e-30), label=name)
        axes[row, 0].set_xscale("log")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_title(f"{label}: densities")
        axes[row, 0].grid(True, which="both", alpha=0.3)

        axes[row, 1].plot(t, te, label="T_e")
        axes[row, 1].plot(t, tmono, label="T_mono")
        axes[row, 1].plot(t, tdiato, label="T_diato")
        axes[row, 1].set_xscale("log")
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_title(f"{label}: temperatures (eV)")
        axes[row, 1].grid(True, which="both", alpha=0.3)

        axes[row, 2].plot(t, eta_ion, color="k")
        axes[row, 2].set_xscale("log")
        axes[row, 2].set_yscale("log")
        axes[row, 2].set_title(f"{label}: ionization ratio")
        axes[row, 2].grid(True, which="both", alpha=0.3)

    for col in range(3):
        axes[1, col].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_3_1_C7_plasma_time_evolution.png"), dpi=220)
    plt.close(fig)


def _power_proxy_from_tracker(tracker: dict) -> dict:
    def sum_last(patterns: list[str]) -> float:
        total = 0.0
        for key, values in tracker.items():
            if not values:
                continue
            if any(p in key for p in patterns):
                total += abs(float(values[-1]))
        return total

    return {
        "ionisation": sum_last(["ion_"]),
        "excitation": sum_last(["exc"]),
        "elastic": sum_last(["ela_", "elastic"]),
        "wall_losses": sum_last(["flux_to_walls", "thermic_diffusion"]),
        "radiation": 0.0,
    }


def figure_8_power_balance(out_dir: Path) -> None:
    records, f107a_map = weather_records()
    labels = ["low_alt", "high_alt"]
    cases = [Case(altitude_km=180.0, area_m2=0.1), Case(altitude_km=230.0, area_m2=0.1)]
    categories = ["ionisation", "excitation", "elastic", "wall_losses", "radiation"]
    values = {c: [] for c in categories}
    for label, case in zip(labels, cases):
        sample = build_sample_from_case(case, records=records, f107a_map=f107a_map)
        res = solve_plasma_sample(sample, power_rf_w=1000.0, fast_mode=False, simulation_name=f"fig8_{label}")
        tracker = res["model"].var_tracker.tracked_variables
        p = _power_proxy_from_tracker(tracker)
        s = max(sum(p.values()), 1e-30)
        for c in categories:
            values[c].append(100.0 * p[c] / s)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(labels), dtype=float)
    for c in categories:
        ax.bar(x, values[c], bottom=bottom, label=c)
        bottom += np.asarray(values[c], dtype=float)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative share [%] (proxy)")
    ax.set_title("Steady-state power-balance proxy")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_3_2_C8_power_balance_proxy.png"), dpi=220)
    plt.close(fig)


def figure_9_prf_sensitivity(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    # Use a near-optimal air-breathing operating point so the plasma stays lit
    # over the full RF-power sweep.
    sample = build_sample_from_case(Case(altitude_km=178.53, area_m2=0.236), records=records, f107a_map=f107a_map)
    p_rf = np.linspace(100.0, 3000.0, 10)
    thrust = []
    isp = []
    eta_mass_proxy = []
    from common import isp_s, mdot_kg_s

    mdot = mdot_kg_s(sample)
    for p in p_rf:
        t = solve_plasma_sample(sample, power_rf_w=float(p), fast_mode=fast_mode, simulation_name=f"fig9_prf_{int(p)}")["thrust_final_N"]
        thrust.append(t)
        isp.append(isp_s(t, mdot))
        eta_mass_proxy.append(t / max(mdot, 1e-30))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(p_rf, thrust)
    ax[0].set_title("Thrust")
    ax[1].plot(p_rf, isp)
    ax[1].set_title("Isp")
    ax[2].plot(p_rf, eta_mass_proxy)
    ax[2].set_title("eta_mass proxy = T/mdot")
    for a in ax:
        a.set_xlabel("P_RF [W]")
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_3_4_C9_prf_sensitivity.png"), dpi=220)
    plt.close(fig)


def figure_10_atomic_oxygen_effect(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    # Use a denser, ignited reference case. Pure end-members lead to solver
    # failures in this chemistry set, so we explore a realistic O-fraction band.
    sample_ref = build_sample_from_case(Case(altitude_km=178.53, area_m2=0.236), records=records, f107a_map=f107a_map)
    n_total = float(sample_ref["N2_m3"] + sample_ref["O2_m3"] + sample_ref["O_m3"] + sample_ref["N_m3"])
    o_fraction = np.linspace(0.2, 0.9, 6)
    te_vals = []
    ne_vals = []

    for frac in o_fraction:
        sample = dict(sample_ref)
        eps = 1e-4 * n_total
        sample["N2_m3"] = float(max((1.0 - frac) * n_total, eps))
        sample["O_m3"] = float(max(frac * n_total, eps))
        sample["O2_m3"] = float(eps)
        sample["N_m3"] = float(eps)
        try:
            res = solve_plasma_sample(
                sample,
                power_rf_w=1000.0,
                fast_mode=fast_mode,
                simulation_name=f"fig10_o_frac_{frac:.2f}",
            )
            states = res["states"]
            species = res["species"]
            ne_vals.append(float(states[-1, 0]))
            te_vals.append(float(states[-1, species.nb]))
        except Exception:
            ne_vals.append(np.nan)
            te_vals.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(o_fraction, te_vals, marker="o")
    axes[0].set_title(r"$T_e$ at equilibrium")
    axes[0].set_ylabel(r"$T_e$ [eV]")
    axes[1].plot(o_fraction, ne_vals, marker="o")
    axes[1].set_yscale("log")
    axes[1].set_title(r"$n_e$ at equilibrium")
    axes[1].set_ylabel(r"$n_e$ [m$^{-3}$]")
    for ax in axes:
        ax.set_xlabel("Atomic oxygen fraction in inflow [-]")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("Fig_3_3_C10_atomic_oxygen_effect.png"), dpi=220)
    plt.close(fig)


def _want(tag: str, selected: set[str]) -> bool:
    return "all" in selected or tag in selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Section C figures: 0D plasma physics.")
    parser.add_argument("--fast", action="store_true", help="Use fast mode when possible.")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["3_1", "3_2", "3_3", "3_4", "all"],
        default=["all"],
        help="Restrict output generation to a subset of report figures.",
    )
    args = parser.parse_args()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only)
    if _want("3_1", selected):
        figure_7_time_evolution(out_dir, fast_mode=bool(args.fast))
    if _want("3_2", selected):
        figure_8_power_balance(out_dir)
    if _want("3_4", selected):
        figure_9_prf_sensitivity(out_dir, fast_mode=bool(args.fast))
    if _want("3_3", selected):
        figure_10_atomic_oxygen_effect(out_dir, fast_mode=bool(args.fast))
    print(f"Saved Section C figures to {out_dir}")


if __name__ == "__main__":
    main()
