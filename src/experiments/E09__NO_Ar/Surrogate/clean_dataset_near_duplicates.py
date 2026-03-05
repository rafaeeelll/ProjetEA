from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = PROJECT_ROOT.joinpath("outputs", "thrust_dataset", "thrust_dataset_msis.json")


def _safe_log10(x: float, floor: float = 1e-30) -> float:
    return float(np.log10(max(float(x), floor)))


def _feature(sample: dict) -> np.ndarray:
    return np.array(
        [
            _safe_log10(sample["N2_m3"]),
            _safe_log10(sample["O2_m3"]),
            _safe_log10(sample["O_m3"]),
            _safe_log10(sample["N_m3"]),
            _safe_log10(sample["intake_area_m2"]),
            _safe_log10(float(sample.get("argon_injection_rate", 0.0)) + 1e14),
        ],
        dtype=float,
    )


def _sample_quality(sample: dict) -> int:
    thrust = sample.get("total_thrust_N", None)
    needs_sim = bool(sample.get("needs_simulation", False))
    if thrust is not None and not needs_sim:
        return 2
    if thrust is None and needs_sim:
        return 1
    return 0


def _neighbor_keys(base_key: tuple[int, ...]) -> list[tuple[int, ...]]:
    dims = len(base_key)
    offsets = np.array(np.meshgrid(*([[-1, 0, 1]] * dims))).T.reshape(-1, dims)
    out = []
    for delta in offsets:
        out.append(tuple(int(base_key[i] + int(delta[i])) for i in range(dims)))
    return out


def _deduplicate(samples: list[dict], feature_tol: float) -> tuple[list[dict], int]:
    order = sorted(range(len(samples)), key=lambda i: (_sample_quality(samples[i]), -i), reverse=True)

    buckets: dict[tuple[int, ...], list[int]] = {}
    kept_samples: list[dict] = []
    kept_feat: list[np.ndarray] = []

    removed = 0
    for idx in order:
        s = samples[idx]
        f = _feature(s)
        key = tuple(np.floor(f / feature_tol).astype(int).tolist())

        duplicate = False
        for nkey in _neighbor_keys(key):
            for kept_idx in buckets.get(nkey, []):
                if np.all(np.abs(f - kept_feat[kept_idx]) <= feature_tol):
                    duplicate = True
                    break
            if duplicate:
                break

        if duplicate:
            removed += 1
            continue

        buckets.setdefault(key, []).append(len(kept_samples))
        kept_samples.append(s)
        kept_feat.append(f)

    return kept_samples, removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove near-duplicate samples from thrust dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Input dataset path.")
    parser.add_argument("--output", type=Path, default=None, help="Output dataset path (default: overwrite input).")
    parser.add_argument(
        "--feature-tol",
        type=float,
        default=0.01,
        help="Max per-feature distance in log10-space to consider samples duplicates.",
    )
    parser.add_argument(
        "--invalidate-thrust",
        action="store_true",
        help="Reset thrust values after cleaning (forces recomputation with current physics).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print stats, do not write file.")
    args = parser.parse_args()

    dataset_path = args.dataset
    output_path = args.output or dataset_path

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    samples = payload.get("samples", [])
    if not samples:
        raise ValueError("Dataset has no samples.")

    cleaned, removed = _deduplicate(samples, feature_tol=float(args.feature_tol))

    if args.invalidate_thrust:
        for s in cleaned:
            s["total_thrust_N"] = None
            s["needs_simulation"] = True
            s.pop("simulation_error", None)

    metadata = payload.setdefault("metadata", {})
    metadata["near_duplicate_feature_tol_log10"] = float(args.feature_tol)
    metadata["near_duplicate_removed_count"] = int(removed)
    metadata["n_points_before_clean"] = int(len(samples))
    metadata["n_points_after_clean"] = int(len(cleaned))
    if args.invalidate_thrust:
        metadata["thrust_invalidated_after_clean"] = True

    print(
        f"Clean dataset: before={len(samples)} after={len(cleaned)} "
        f"removed={removed} tol={float(args.feature_tol):.4f}"
    )
    if args.invalidate_thrust:
        print("All thrust values invalidated (needs_simulation=True).")

    if args.dry_run:
        return

    payload["samples"] = cleaned
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
