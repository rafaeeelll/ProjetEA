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
        te = states[:, species.nb]
        tmono = states[:, species.nb + 1]
        tdiato = states[:, species.nb + 2]
        ne = states[:, 0]
        n_total = np.maximum(np.sum(states[:, : species.nb], axis=1), 1e-30)
        eta_ion = ne / n_total

        for name in ["e", "N2", "O", "N2+", "O+"]:
            idx = _species_idx(species, name)
            if idx is not None:
                axes[row, 0].plot(t, states[:, idx], label=name)
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_title(f"{label}: densities")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, te, label="T_e")
        axes[row, 1].plot(t, tmono, label="T_mono")
        axes[row, 1].plot(t, tdiato, label="T_diato")
        axes[row, 1].set_title(f"{label}: temperatures (eV)")
        axes[row, 1].grid(True, alpha=0.3)

        axes[row, 2].plot(t, eta_ion, color="k")
        axes[row, 2].set_title(f"{label}: ionization ratio")
        axes[row, 2].grid(True, alpha=0.3)

    for col in range(3):
        axes[1, col].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("C7_plasma_time_evolution.png"), dpi=220)
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
    fig.savefig(out_dir.joinpath("C8_power_balance_proxy.png"), dpi=220)
    plt.close(fig)


def figure_9_prf_sensitivity(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    sample = build_sample_from_case(Case(altitude_km=190.0, area_m2=0.1), records=records, f107a_map=f107a_map)
    p_rf = np.linspace(100.0, 3000.0, 16)
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
    fig.savefig(out_dir.joinpath("C9_prf_sensitivity.png"), dpi=220)
    plt.close(fig)


def figure_10_atomic_oxygen_effect(out_dir: Path, fast_mode: bool) -> None:
    records, f107a_map = weather_records()
    sample_ref = build_sample_from_case(Case(altitude_km=210.0, area_m2=0.1), records=records, f107a_map=f107a_map)
    sample_no_o = dict(sample_ref)
    sample_no_o["O_m3"] = 0.0

    res_ref = solve_plasma_sample(sample_ref, power_rf_w=1000.0, fast_mode=fast_mode, simulation_name="fig10_with_O")
    res_no_o = solve_plasma_sample(sample_no_o, power_rf_w=1000.0, fast_mode=fast_mode, simulation_name="fig10_no_O")

    def summary(res):
        states = res["states"]
        species = res["species"]
        ne = float(states[-1, 0])
        n_total = float(np.sum(states[-1, : species.nb]))
        te = float(states[-1, species.nb])
        ratio = ne / max(n_total, 1e-30)
        ions = {}
        for ion in ["N2+", "O+", "N+", "O2+"]:
            try:
                ions[ion] = float(states[-1, species.get_specie_by_name(ion).index])
            except Exception:
                ions[ion] = 0.0
        return ratio, te, ions

    eta_ref, te_ref, ions_ref = summary(res_ref)
    eta_no, te_no, ions_no = summary(res_no_o)

    labels = ["with O", "without O"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(labels, [eta_ref, eta_no])
    axes[0].set_title("Ionization ratio")
    axes[1].bar(labels, [te_ref, te_no])
    axes[1].set_title("Te [eV]")
    ions_names = ["N2+", "O+", "N+", "O2+"]
    x = np.arange(len(ions_names))
    width = 0.35
    axes[2].bar(x - width / 2, [ions_ref[k] for k in ions_names], width=width, label="with O")
    axes[2].bar(x + width / 2, [ions_no[k] for k in ions_names], width=width, label="without O")
    axes[2].set_xticks(x, ions_names)
    axes[2].set_yscale("log")
    axes[2].set_title("Ion composition")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("C10_atomic_oxygen_effect.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section C figures: 0D plasma physics.")
    parser.add_argument("--fast", action="store_true", help="Use fast mode when possible.")
    args = parser.parse_args()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_7_time_evolution(out_dir, fast_mode=bool(args.fast))
    figure_8_power_balance(out_dir)
    figure_9_prf_sensitivity(out_dir, fast_mode=bool(args.fast))
    figure_10_atomic_oxygen_effect(out_dir, fast_mode=bool(args.fast))
    print(f"Saved Section C figures to {out_dir}")


if __name__ == "__main__":
    main()
