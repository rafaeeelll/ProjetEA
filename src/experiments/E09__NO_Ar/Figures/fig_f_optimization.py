from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, beta_by_species, collection_efficiency


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOG_PATH = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "active_learning_log.json")


def _load_log(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def figure_17_convergence(out_dir: Path, payload: dict) -> None:
    rows = payload.get("iterations", [])
    if not rows:
        raise ValueError("No iterations in optimization log.")
    it = np.array([r["iteration"] for r in rows], dtype=int)
    j_lcb = np.array([r.get("j_lcb_N", np.nan) for r in rows], dtype=float)
    j_mu = np.array([r.get("j_mu_N", np.nan) for r in rows], dtype=float)
    sig = np.array([r.get("sigma_bottleneck_N", np.nan) for r in rows], dtype=float)
    alt = np.array([r.get("design_altitude_km", np.nan) for r in rows], dtype=float)
    area = np.array([r.get("design_intake_area_m2", np.nan) for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(it, j_lcb, label="J_lcb")
    axes[0, 0].plot(it, j_mu, label="J_mu")
    axes[0, 0].set_ylabel("N")
    axes[0, 0].set_title("Objective convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(it, sig, color="tab:orange")
    axes[0, 1].set_ylabel("N")
    axes[0, 1].set_title("sigma bottleneck")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(it, alt, color="tab:green")
    axes[1, 0].set_ylabel("km")
    axes[1, 0].set_title("altitude*")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(it, area, color="tab:red")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("m²")
    axes[1, 1].set_title("A_intake*")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir.joinpath("F17_active_learning_convergence.png"), dpi=220)
    plt.close(fig)


def figure_18_summary_table(out_dir: Path, payload: dict) -> None:
    final_summary = payload.get("final_summary", {})
    best_design = final_summary.get("best_design", {})
    if not best_design:
        # fallback to best iteration row
        rows = payload.get("iterations", [])
        if rows:
            best_row = max(rows, key=lambda r: float(r.get("j_lcb_N", -1e99)))
            best_design = {
                "altitude_km": best_row.get("design_altitude_km", np.nan),
                "intake_area_m2": best_row.get("design_intake_area_m2", np.nan),
                "argon_injection_rate": best_row.get("design_argon_injection_rate", 0.0),
                "thrust_N": best_row.get("true_thrust_bottleneck_N", np.nan),
                "margin_N": best_row.get("j_mu_N", np.nan),
            }

    altitude = float(best_design.get("altitude_km", np.nan))
    area = float(best_design.get("intake_area_m2", np.nan))
    argon = float(best_design.get("argon_injection_rate", 0.0))
    eta_eff = float(collection_efficiency(area)) if np.isfinite(area) else np.nan
    u_orb = float(np.sqrt(3.986004418e14 / (6371e3 + max(altitude, 1.0) * 1e3))) if np.isfinite(altitude) else np.nan
    beta = beta_by_species(
        orbital_speed_m_s=u_orb if np.isfinite(u_orb) else 7800.0,
        t_wall_k=300.0,
        a_intake_m2=area if np.isfinite(area) else 0.1,
        eta_collection=eta_eff if np.isfinite(eta_eff) else 0.4,
    )

    thrust = float(final_summary.get("true_thrust_at_min_margin_N", best_design.get("thrust_N", np.nan)))
    margin = float(final_summary.get("true_min_margin_N", best_design.get("margin_N", np.nan)))
    drag = thrust - margin if np.isfinite(thrust) and np.isfinite(margin) else np.nan
    mdot = float(final_summary.get("true_mdot_at_min_margin_kg_s", np.nan))
    isp = float(final_summary.get("true_isp_at_min_margin_s", np.nan))
    td_ratio = thrust / max(drag, 1e-30) if np.isfinite(thrust) and np.isfinite(drag) else np.nan

    rows = [
        ("Altitude", f"{altitude:.3f} km"),
        ("A_intake", f"{area:.4f} m²"),
        ("Argon rate", f"{argon:.3e} part/s"),
        ("eta_c effectif", f"{eta_eff:.3f}"),
        ("beta (N2/O)", f"{beta['N2']:.2f} / {beta['O']:.2f}"),
        ("Thrust", f"{thrust*1e3:.3f} mN" if np.isfinite(thrust) else "n/a"),
        ("Drag", f"{drag*1e3:.3f} mN" if np.isfinite(drag) else "n/a"),
        ("Margin", f"{margin*1e3:.3f} mN" if np.isfinite(margin) else "n/a"),
        ("Isp", f"{isp:.1f} s" if np.isfinite(isp) else "n/a"),
        ("mdot", f"{mdot:.3e} kg/s" if np.isfinite(mdot) else "n/a"),
        ("T/D ratio", f"{td_ratio:.3f}" if np.isfinite(td_ratio) else "n/a"),
        ("P_RF", "1000 W"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Parameter", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title("Optimal design summary", pad=12)
    fig.tight_layout()
    fig.savefig(out_dir.joinpath("F18_optimal_design_table.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Section F figures: optimization.")
    parser.add_argument("--log-path", type=Path, default=LOG_PATH, help="Path to active learning log JSON.")
    args = parser.parse_args()

    payload = _load_log(args.log_path)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_17_convergence(out_dir, payload)
    figure_18_summary_table(out_dir, payload)
    print(f"Saved Section F figures to {out_dir}")


if __name__ == "__main__":
    main()
