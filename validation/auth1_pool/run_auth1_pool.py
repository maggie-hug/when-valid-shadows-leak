"""Evaluate the predeclared authentication-one position-pool ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "validation" / "support_decoder"))
from run_repeated_linkage import (  # noqa: E402
    READINGS, build_channel, build_position_overlaps, greedy_schedule,
    load_gallery, sample_observations, stable_seed, success_bound, summarize,
)
from run_support_ablation import decode, metric_values, METRICS  # noqa: E402

SEED = 20260830
TRIALS = 1000
BUDGETS = (8, 12)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "auth1_pool")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("trials.csv", "observations.npz", "summary.json", "run_manifest.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Preserve retained evidence: choose a new --output-dir")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    dataset_root = args.dataset_root.resolve()
    input_paths = [
        Path(__file__).resolve(), HERE / "README.md",
        ROOT / "scripts" / "run_repeated_linkage.py",
        ROOT / "validation" / "support_decoder" / "run_support_ablation.py",
        ROOT / "data" / "branch_ablation.csv",
    ]
    input_hashes = {str(path): sha256(path) for path in input_paths}
    secrets, auth, mask_name = load_gallery(dataset_root, "test")
    eligible = np.flatnonzero(auth == 1).astype(np.int32)
    if secrets.shape != (200, 4096) or mask_name != "100075.jpg" or len(eligible) != 1982:
        raise AssertionError("Gallery/mask differs from the declared protocol")
    gallery_paths = sorted((dataset_root / "images" / "test").glob("*.jpg"))
    image_paths = gallery_paths + [dataset_root / "images" / "train" / mask_name]
    dataset_hashes = {str(path.relative_to(dataset_root)): sha256(path)
                      for path in image_paths}
    with (ROOT / "data" / "branch_ablation.csv").open(newline="", encoding="utf-8") as handle:
        prior_bounds = {
            (int(row["participant"]), row["reading"], int(row["budget"])): float(row["bound"])
            for row in csv.DictReader(handle) if row["candidate_pool"] == "auth1"
        }

    rows, strata, observed_cache = [], [], {}
    per_budget = {budget: [] for budget in BUDGETS}
    for participant in (1, 2):
        for reading, stop in READINGS.items():
            channels = build_channel(stop, participant)
            rho, left, right = build_position_overlaps(secrets, auth, channels)
            full_schedule = greedy_schedule(rho, max(BUDGETS), eligible)
            if len(set(full_schedule)) != max(BUDGETS) or not np.all(auth[full_schedule] == 1):
                raise AssertionError("Schedule contains duplicates or ineligible positions")
            bounds = {}
            for budget in BUDGETS:
                bounds[budget] = success_bound(
                    full_schedule[:budget], rho, left, right, len(secrets))
                if not np.isclose(bounds[budget], prior_bounds[participant, reading, budget],
                                  rtol=0.0, atol=1e-12):
                    raise AssertionError("Recomputed certificate differs from prior branch ablation")
            del rho, left, right
            cdfs = {bit: np.cumsum(channel, axis=1) for bit, channel in channels.items()}
            for cdf in cdfs.values():
                cdf[:, -1] = 1.0
            sorted_positions = np.sort(full_schedule)
            key = f"p{participant}_{reading}"
            cache = np.empty((TRIALS, max(BUDGETS), len(secrets)), dtype=np.uint8)
            values = {budget: [] for budget in BUDGETS}
            for trial in range(TRIALS):
                rng = np.random.default_rng(stable_seed(
                    SEED, "auth1_pool", f"participant{participant}", reading, "trial", trial, "test"))
                observations = sample_observations(sorted_positions, secrets, auth, cdfs, rng)
                for offset, position in enumerate(full_schedule):
                    cache[trial, offset] = observations[int(position)]
                for budget in BUDGETS:
                    counts = decode(full_schedule[:budget], cache[trial, :budget],
                                    secrets, auth, channels)
                    row = {"budget": budget, "participant": participant, "reading": reading,
                           "trial": trial, "targets": len(secrets), **counts}
                    rows.append(row)
                    values[budget].append(metric_values(row))
            for budget in BUDGETS:
                sample = np.asarray(values[budget])
                per_budget[budget].append(sample)
                item = {"budget": budget, "participant": participant, "reading": reading,
                        "positions": full_schedule[:budget].tolist(),
                        "certificate": bounds[budget],
                        **{metric: summarize(sample[:, i]) for i, metric in enumerate(METRICS)}}
                item["relative_map_gain_over_support"] = (
                    item["map_success"]["mean"] / item["support_success"]["mean"] - 1.0)
                strata.append(item)
            observed_cache[key] = cache
            print(f"Completed {key}: {TRIALS} full-gallery trials, budgets {BUDGETS}", flush=True)

    aggregate = []
    for budget in BUDGETS:
        if len(per_budget[budget]) != 4:
            raise AssertionError("Expected exactly four equally weighted conditions")
        trial_means = np.mean(np.stack(per_budget[budget]), axis=0)
        item = {"budget": budget,
                "certificate_mean": float(np.mean([item["certificate"] for item in strata
                                                   if item["budget"] == budget])),
                **{metric: summarize(trial_means[:, i]) for i, metric in enumerate(METRICS)}}
        item["relative_map_gain_over_support"] = (
            item["map_success"]["mean"] / item["support_success"]["mean"] - 1.0)
        aggregate.append(item)
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "observations.npz", **observed_cache)
    summary = {"status": "completed", "candidate_branch": 1,
               "candidate_positions": len(eligible), "gallery_size": len(secrets),
               "trials_per_condition": TRIALS, "conditions_per_budget": 4,
               "strata": strata, "four_condition_means": aggregate}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if {str(path): sha256(path) for path in input_paths} != input_hashes:
        raise AssertionError("An experiment input changed while the run was in progress")
    manifest = {
        "status": "completed", "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
        "dataset_root": str(dataset_root), "mask": mask_name,
        "gallery_order": [path.name for path in gallery_paths],
        "dataset_sha256": dataset_hashes, "input_sha256": input_hashes,
        "output_sha256": {name: sha256(output / name) for name in names[:-1]},
        "seed": SEED,
        "seed_parts": ["base", "auth1_pool", "participant{1|2}", "reading", "trial", "trial_id", "test"],
        "budgets": list(BUDGETS), "trials_per_condition": TRIALS,
        "evaluation_type": "simulation_only", "ground_truth": "generating gallery image index",
        "observation_source": "enumerated-channel sampling",
        "tie_rule": "MAP: first maximum of accumulated float64 log scores; support: lowest surviving candidate index",
        "scope": "auth1-only Greedy schedules on one fixed 200-image gallery; nested budgets share observations",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"four_condition_means": aggregate}, indent=2))


if __name__ == "__main__":
    main()
