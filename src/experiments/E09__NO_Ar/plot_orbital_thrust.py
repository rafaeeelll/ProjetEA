import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    log_folder_path = (
        Path(__file__).resolve().parent.parent.parent.parent.joinpath(
            "outputs", "logs_for_thrust_by_altitude"
        )
    )
    json_path = log_folder_path.joinpath("thrust_vs_orbit.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing file: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)

    angles = np.asarray(results.get("angles_rad", []), dtype=float)
    with_argon = results.get("with_argon", {})
    without_argon = results.get("without_argon", {})

    y_with_ion = np.asarray(with_argon.get("ion_thrust_N", []), dtype=float)
    y_with_total = np.asarray(with_argon.get("total_thrust_N", []), dtype=float)
    y_wo_ion = np.asarray(without_argon.get("ion_thrust_N", []), dtype=float)
    y_wo_total = np.asarray(without_argon.get("total_thrust_N", []), dtype=float)

    if angles.size == 0:
        n = max(y_with_ion.size, y_with_total.size, y_wo_ion.size, y_wo_total.size)
        if n == 0:
            raise ValueError("No data found in thrust_vs_orbit.json.")
        angles = np.linspace(0.0, 2.0 * np.pi, n)

    plt.figure(figsize=(8, 5))
    if y_with_ion.size:
        plt.plot(angles, y_with_ion, marker="o", label="Ion thrust (Ar)")
    if y_with_total.size:
        plt.plot(angles, y_with_total, marker="s", label="Total thrust (Ar)")
    if y_wo_ion.size:
        plt.plot(
            angles[: y_wo_ion.size],
            y_wo_ion,
            marker="o",
            linestyle="--",
            label="Ion thrust (no Ar)",
        )
    if y_wo_total.size:
        plt.plot(
            angles[: y_wo_total.size],
            y_wo_total,
            marker="s",
            linestyle="--",
            label="Total thrust (no Ar)",
        )
    plt.xlabel("Angle orbital (rad)")
    plt.ylabel("Thrust (N)")
    plt.title("Thrust vs Angle (from JSON)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_folder_path.joinpath("thrust_vs_orbit_from_json.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
