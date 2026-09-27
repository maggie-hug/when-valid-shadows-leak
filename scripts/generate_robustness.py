from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_repeated_linkage import (
    DEFAULT_SEED,
    READINGS,
    build_channel,
    build_position_overlaps,
    greedy_schedule,
    load_gallery,
    preprocess,
    random_schedules,
    stable_seed,
    success_bound,
)


GALLERY_SIZES = (25, 50, 100, 200)
SUBSET_REPEATS = 10
MASK_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate gallery-size and public-mask robustness bounds."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BSDS500 data directory containing images/train and images/test",
    )
    parser.add_argument("--participant", type=int, choices=(1, 2), default=1)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--random-schedules", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "robustness",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset_indices(total: int, size: int, repeat: int, seed: int) -> np.ndarray:
    if size == total:
        return np.arange(total, dtype=np.int32)
    rng = np.random.default_rng(
        stable_seed(seed, "robust_gallery_subset", size, repeat)
    )
    return np.sort(rng.choice(total, size, replace=False)).astype(np.int32)


def summarize(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.min(array)), float(np.max(array))


def bound_pair(
    secrets: np.ndarray,
    auth: np.ndarray,
    channels: dict[int, np.ndarray],
    budget: int,
    schedule_count: int,
    seed: int,
) -> tuple[float, float]:
    rho, pair_left, pair_right = build_position_overlaps(secrets, auth, channels)
    eligible = np.flatnonzero(auth == 0).astype(np.int32)
    greedy = greedy_schedule(rho, budget, eligible)
    greedy_bound = success_bound(
        greedy, rho, pair_left, pair_right, len(secrets)
    )
    random = random_schedules(auth, schedule_count, seed, budget)
    random_bounds = [
        success_bound(schedule, rho, pair_left, pair_right, len(secrets))
        for schedule in random
    ]
    return greedy_bound, float(np.mean(random_bounds))


def gallery_robustness(
    secrets: np.ndarray,
    auth: np.ndarray,
    channels_by_reading: dict[str, dict[int, np.ndarray]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reading, channels in channels_by_reading.items():
        for size in GALLERY_SIZES:
            repeats = 1 if size == len(secrets) else SUBSET_REPEATS
            greedy_values: list[float] = []
            random_values: list[float] = []
            for repeat in range(repeats):
                subset = subset_indices(len(secrets), size, repeat, args.seed)
                greedy_bound, random_bound = bound_pair(
                    secrets[subset],
                    auth,
                    channels,
                    args.budget,
                    args.random_schedules,
                    stable_seed(args.seed, "gallery_random", size, repeat),
                )
                greedy_values.append(greedy_bound)
                random_values.append(random_bound)
            for method, values in (
                ("greedy", greedy_values),
                ("matched_random_mean", random_values),
            ):
                mean, minimum, maximum = summarize(values)
                rows.append(
                    {
                        "reading": reading,
                        "participant": args.participant,
                        "gallery_size": size,
                        "method": method,
                        "budget": args.budget,
                        "subsets": repeats,
                        "bound_mean": mean,
                        "bound_min": minimum,
                        "bound_max": maximum,
                    }
                )
    return rows


def public_masks(dataset_root: Path) -> list[tuple[str, np.ndarray]]:
    paths = sorted((dataset_root / "images" / "train").glob("*.jpg"))[:MASK_COUNT]
    if len(paths) != MASK_COUNT:
        raise RuntimeError(f"expected {MASK_COUNT} training images, found {len(paths)}")
    masks: list[tuple[str, np.ndarray]] = []
    for path in paths:
        image = preprocess(path)
        mask = (image > float(np.median(image))).astype(np.uint8).reshape(-1)
        if not (np.any(mask == 0) and np.any(mask == 1)):
            raise AssertionError(f"degenerate public mask: {path.name}")
        masks.append((path.name, mask))
    return masks


def mask_robustness(
    secrets: np.ndarray,
    masks: list[tuple[str, np.ndarray]],
    channels_by_reading: dict[str, dict[int, np.ndarray]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reading, channels in channels_by_reading.items():
        for mask_name, auth in masks:
            greedy_bound, random_bound = bound_pair(
                secrets,
                auth,
                channels,
                args.budget,
                args.random_schedules,
                stable_seed(args.seed, "mask_random", mask_name),
            )
            rows.append(
                {
                    "reading": reading,
                    "participant": args.participant,
                    "mask": mask_name,
                    "auth0_count": int(np.sum(auth == 0)),
                    "auth1_count": int(np.sum(auth == 1)),
                    "budget": args.budget,
                    "greedy_bound": greedy_bound,
                    "matched_random_bound_mean": random_bound,
                    "gain": greedy_bound - random_bound,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    secrets, auth, _ = load_gallery(dataset_root, "test")
    channels_by_reading = {
        reading: build_channel(coefficient_stop, args.participant)
        for reading, coefficient_stop in READINGS.items()
    }

    gallery_rows = gallery_robustness(
        secrets, auth, channels_by_reading, args
    )
    mask_rows = mask_robustness(
        secrets, public_masks(dataset_root), channels_by_reading, args
    )
    write_csv(output_dir / "robustness_summary.csv", gallery_rows)
    write_csv(output_dir / "auth_mask_robustness.csv", mask_rows)
    print(f"wrote {len(gallery_rows)} gallery rows and {len(mask_rows)} mask rows")


if __name__ == "__main__":
    main()
