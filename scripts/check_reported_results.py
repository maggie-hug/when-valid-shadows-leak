"""Recompute headline paper statistics from the published numerical records."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READINGS = ("full_field", "byte_domain")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def decoder_summary(folder: str) -> dict[str, float]:
    rows = [row for row in read_csv(ROOT / "validation" / folder / "results" / "trials.csv")
            if int(row["budget"]) == 12]
    keys = {(int(row["participant"]), row["reading"], int(row["trial"])) for row in rows}
    expected = {(p, d, t) for p in (1, 2) for d in READINGS for t in range(1000)}
    require(keys == expected and len(rows) == len(expected), f"Incomplete {folder} trials")
    for row in rows:
        require(int(row["targets"]) == 200, "Wrong gallery size")
        require(0 <= int(row["support_correct"]) <= 200, "Invalid support count")
        require(0 <= int(row["map_correct"]) <= 200, "Invalid MAP count")
    total = sum(int(row["targets"]) for row in rows)
    return {name: 100 * sum(int(row[field]) for row in rows) / total
            for name, field in (("map", "map_correct"), ("support_only", "support_correct"),
                                ("unique_survivor", "unique_targets"))}


def main() -> None:
    condition_trials = []
    for participant in (1, 2):
        rows = read_csv(ROOT / "data" / f"repeated_shadow_trials_p{participant}_b12.csv")
        for reading in READINGS:
            selected = [row for row in rows if row["method"] == "greedy" and row["reading"] == reading]
            require(len(selected) == 1000, "Missing greedy trial")
            by_trial = {int(row["trial"]): float(row["top1"]) for row in selected}
            require(set(by_trial) == set(range(1000)), "Missing or duplicate trial ID")
            values = [by_trial[t] for t in range(1000)]
            require(all(0 <= value <= 1 for value in values), "Invalid success fraction")
            condition_trials.append(values)
    trial_means = [statistics.mean(values) for values in zip(*condition_trials)]
    top1 = 100 * statistics.mean(trial_means)
    half_width = 100 * 1.96 * statistics.stdev(trial_means) / math.sqrt(len(trial_means))
    bounds = [float(row["bound"]) for row in read_csv(ROOT / "data" / "bound_summary.csv")
              if row["method"] == "greedy" and int(row["budget"]) == 12]
    require(len(bounds) == 4, "Expected four certificate conditions")
    minimum = 100 * min(bounds)
    average = 100 * statistics.mean(bounds)
    zero_bit = decoder_summary("support_decoder")
    one_bit = decoder_summary("auth1_pool")
    require(round(top1, 2) == 98.07, "Headline identification rate differs")
    require(round(minimum, 2) == 95.88 and round(average, 2) == 95.96, "Certificate differs")
    require(round(zero_bit["support_only"], 2) == 97.99, "Zero-bit decoder rate differs")
    require(round(one_bit["map"], 2) == 6.76 and round(one_bit["support_only"], 2) == 0.54,
            "One-bit decoder rates differ")
    print(json.dumps({"status": "all_passed", "units": "percent",
                      "top1_mean": top1, "top1_ci95": [top1 - half_width, top1 + half_width],
                      "certificate_mean": average, "certificate_minimum": minimum,
                      "zero_bit": zero_bit, "one_bit": one_bit}, indent=2))


if __name__ == "__main__":
    main()
