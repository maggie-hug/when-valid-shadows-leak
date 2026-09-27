"""Replay retained observations and compare MAP with support-only decoding."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_repeated_linkage import (  # noqa: E402
    READINGS, build_channel, load_gallery, random_schedules, stable_seed, summarize,
)

SEED = 20260830
TRIALS = 1000
BUDGETS = (8, 12)
METRICS = ("map_success", "support_success", "unique_fraction", "mean_survivors",
           "map_minus_support")


def decode(schedule, observed, secrets, auth, channels):
    """Candidate rows, target columns; target identity is never used to rank."""
    count = len(secrets)
    scores = np.zeros((count, count), dtype=np.float64)
    possible = np.ones((count, count), dtype=bool)
    for offset, position in enumerate(schedule):
        masses = channels[int(auth[position])][
            secrets[:, position, None], observed[offset][None, :]
        ]
        possible &= masses > 0.0
        log_masses = np.full(masses.shape, -np.inf)
        np.log(masses, out=log_masses, where=masses > 0.0)
        scores += log_masses
    truth = np.arange(count)
    if not np.all(possible[truth, truth]):
        raise AssertionError("The true candidate was excluded by its own support")
    if not np.array_equal(possible, np.isfinite(scores)):
        raise AssertionError("Support and finite log likelihood disagree")
    survivors = possible.sum(axis=0)
    map_prediction = scores.argmax(axis=0)
    support_prediction = possible.argmax(axis=0)
    return {
        "map_correct": int(np.count_nonzero(map_prediction == truth)),
        "support_correct": int(np.count_nonzero(support_prediction == truth)),
        "unique_targets": int(np.count_nonzero(survivors == 1)),
        "survivor_total": int(survivors.sum()),
    }


def metric_values(row):
    count = row["targets"]
    return np.array([
        row["map_correct"] / count,
        row["support_correct"] / count,
        row["unique_targets"] / count,
        row["survivor_total"] / count,
        (row["map_correct"] - row["support_correct"]) / count,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "support_decoder")
    args = parser.parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    secrets, auth, mask_name = load_gallery(args.dataset_root.resolve(), "test")
    if secrets.shape != (200, 4096) or mask_name != "100075.jpg":
        raise AssertionError("Gallery/mask differs from the declared experiment")
    rows, strata, observed_cache, input_paths = [], [], {}, []
    per_budget = {budget: [] for budget in BUDGETS}
    replay_matches = 0

    for participant in (1, 2):
        for reading, stop in READINGS.items():
            channels = build_channel(stop, participant)
            cdfs = {bit: np.cumsum(channel, axis=1)
                    for bit, channel in channels.items()}
            for cdf in cdfs.values():
                cdf[:, -1] = 1.0
            for budget in BUDGETS:
                summary_path = ROOT / "data" / f"repeated_shadow_summary_p{participant}_b{budget}.json"
                trial_path = ROOT / "data" / f"repeated_shadow_trials_p{participant}_b{budget}.csv"
                retained = json.loads(summary_path.read_text(encoding="utf-8"))
                for key, expected in (("trials", TRIALS), ("gallery_size", 200),
                                      ("participant", participant), ("budget", budget),
                                      ("candidate_branch", 0), ("gallery_split", "test"),
                                      ("authentication_source", mask_name)):
                    if retained[key] != expected:
                        raise AssertionError(f"Unexpected retained setting {key}")
                fixed = retained["readings"][reading]
                greedy, static, mi = [np.array(fixed[name + "_positions"], dtype=np.int32)
                                      for name in ("greedy", "static", "mi_rank")]
                if len(greedy) != budget or len(np.unique(greedy)) != budget or np.any(auth[greedy]):
                    raise AssertionError("Invalid saved Greedy schedule")
                random = random_schedules(auth, retained["random_schedules"], SEED, budget)
                with trial_path.open(newline="", encoding="utf-8") as handle:
                    original_rows = [row for row in csv.DictReader(handle)
                                     if row["reading"] == reading and row["method"] == "greedy"]
                retained_values = {int(row["trial"]): float(row["top1"]) for row in original_rows}
                if len(original_rows) != TRIALS or set(retained_values) != set(range(TRIALS)):
                    raise AssertionError("Retained trial IDs are incomplete or duplicated")
                input_paths.extend([str(summary_path), str(trial_path)])
                key = f"p{participant}_{reading}_b{budget}"
                cached = np.empty((TRIALS, budget, len(secrets)), dtype=np.uint8)
                values = []
                for trial in range(TRIALS):
                    union = np.unique(np.concatenate((greedy, static, mi, random[trial % len(random)])))
                    rng = np.random.default_rng(stable_seed(
                        SEED, f"participant{participant}", reading, "trial", trial, "test"))
                    # Consume the original draws at all union positions in their
                    # original sorted order, including positions not decoded here.
                    draws = rng.random((len(union), len(secrets)))
                    observed = cached[trial]
                    for offset, position in enumerate(greedy):
                        union_index = int(np.searchsorted(union, position))
                        cdf_rows = cdfs[int(auth[position])][secrets[:, position]]
                        observed[offset] = np.argmax(
                            draws[union_index, :, None] <= cdf_rows, axis=1).astype(np.uint8)
                    counts = decode(greedy, observed, secrets, auth, channels)
                    if counts["map_correct"] / len(secrets) != retained_values[trial]:
                        raise AssertionError(f"MAP replay mismatch at {key}, trial {trial}")
                    replay_matches += 1
                    row = {"budget": budget, "participant": participant, "reading": reading,
                           "trial": trial, "targets": len(secrets), **counts}
                    rows.append(row)
                    values.append(metric_values(row))
                values = np.asarray(values)
                per_budget[budget].append(values)
                item = {"budget": budget, "participant": participant, "reading": reading,
                        "positions": greedy.tolist(),
                        **{metric: summarize(values[:, i]) for i, metric in enumerate(METRICS)}}
                item["relative_map_gain_over_support"] = (
                    item["map_success"]["mean"] / item["support_success"]["mean"] - 1.0)
                strata.append(item)
                observed_cache[key] = cached
                print(f"Completed {key}: MAP replay matched {TRIALS} trials", flush=True)

    aggregate = []
    for budget in BUDGETS:
        if len(per_budget[budget]) != 4:
            raise AssertionError("Expected exactly four equally weighted conditions")
        trial_means = np.mean(np.stack(per_budget[budget]), axis=0)
        item = {"budget": budget,
                **{metric: summarize(trial_means[:, i]) for i, metric in enumerate(METRICS)}}
        item["relative_map_gain_over_support"] = (
            item["map_success"]["mean"] / item["support_success"]["mean"] - 1.0)
        aggregate.append(item)
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "observations.npz", **observed_cache)
    summary = {"status": "completed", "trials_per_condition": TRIALS,
               "gallery_size": len(secrets), "conditions_per_budget": 4,
               "map_replay_matches": replay_matches, "strata": strata,
               "four_condition_means": aggregate}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {"status": "completed", "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.perf_counter() - started,
                "python": platform.python_version(), "numpy": np.__version__,
                "pillow": PIL.__version__, "dataset_root": str(args.dataset_root.resolve()),
                "mask": mask_name, "seed": SEED, "budgets": list(BUDGETS),
                "trials_per_condition": TRIALS, "input_paths": sorted(set(input_paths)),
                "protocol": str(Path(__file__).resolve().parent / "README.md"),
                "script": str(Path(__file__).resolve()),
                "tie_rule": "MAP: first maximum of accumulated float64 log scores; support: lowest surviving candidate index",
                "observation_source": "replayed enumerated-channel sampling",
                "scope": "decoder ablation at unchanged probability-selected Greedy positions"}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"map_replay_matches": replay_matches,
                      "four_condition_means": aggregate}, indent=2))


if __name__ == "__main__":
    main()
