from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image


FIELD = 257
IMAGE_SIZE = 64
EXPECTED_GALLERY_SIZES = {"test": 200, "val": 100}
DEFAULT_BUDGET = 8
POSITION_CHUNK = 64
DEFAULT_TRIALS = 200
DEFAULT_RANDOM_SCHEDULES = 1000
DEFAULT_SEED = 20260830

PATTERNS = {
    0: (((0, 0), 0.5), ((1, 1), 0.5)),
    1: (
        ((0, 0), 1.0 / 6.0),
        ((1, 1), 1.0 / 6.0),
        ((0, 1), 1.0 / 3.0),
        ((1, 0), 1.0 / 3.0),
    ),
}
READINGS = {"full_field": 257, "byte_domain": 256}


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def parity4(value: int) -> int:
    return (value ^ (value >> 1) ^ (value >> 2) ^ (value >> 3)) & 1


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def load_gallery(
    dataset_root: Path, split: str = "test"
) -> tuple[np.ndarray, np.ndarray, str]:
    gallery_paths = sorted((dataset_root / "images" / split).glob("*.jpg"))
    expected = EXPECTED_GALLERY_SIZES[split]
    if len(gallery_paths) != expected:
        raise RuntimeError(
            f"expected {expected} {split} images, found {len(gallery_paths)}"
        )
    train_paths = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    if not train_paths:
        raise RuntimeError("no training image is available for the public mask")
    secrets = np.stack([preprocess(path) for path in gallery_paths]).reshape(
        expected, -1
    )
    auth_image = preprocess(train_paths[0])
    auth = (auth_image > float(np.median(auth_image))).astype(np.uint8).reshape(-1)
    return secrets, auth, train_paths[0].name


def build_channel(coefficient_stop: int, participant: int) -> dict[int, np.ndarray]:
    if participant not in (1, 2):
        raise ValueError("participant must be 1 or 2")
    channels: dict[int, np.ndarray] = {}
    for bit in (0, 1):
        channel = np.zeros((256, 256), dtype=np.float64)
        for secret in range(256):
            for pattern, weight in PATTERNS[bit]:
                accepted: list[int] = []
                for coefficient in range(coefficient_stop):
                    first = (secret + coefficient) % FIELD
                    second = (secret + 2 * coefficient) % FIELD
                    if first >= 256 or second >= 256:
                        continue
                    if (parity4(first), parity4(second)) == pattern:
                        accepted.append(first if participant == 1 else second)
                if not accepted:
                    raise AssertionError(
                        f"empty accepted set: bit={bit}, secret={secret}, "
                        f"pattern={pattern}, stop={coefficient_stop}"
                    )
                mass = weight / len(accepted)
                for observed in accepted:
                    channel[secret, observed] += mass
        if not np.allclose(channel.sum(axis=1), 1.0, atol=1e-12):
            raise AssertionError("channel rows are not normalized")
        channels[bit] = channel
    return channels


def build_position_overlaps(
    secrets: np.ndarray, auth: np.ndarray, channels: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(len(secrets), k=1)
    overlaps = {
        bit: np.sqrt(channels[bit]) @ np.sqrt(channels[bit]).T for bit in (0, 1)
    }
    rho = np.empty((secrets.shape[1], len(left)), dtype=np.float32)
    for bit in (0, 1):
        positions = np.flatnonzero(auth == bit)
        for start in range(0, len(positions), POSITION_CHUNK):
            chunk = positions[start : start + POSITION_CHUNK]
            values = secrets[:, chunk].T
            rho[chunk] = overlaps[bit][values[:, left], values[:, right]].astype(
                np.float32
            )
    return rho, left.astype(np.int32), right.astype(np.int32)


def greedy_schedule(
    rho: np.ndarray, budget: int, eligible: np.ndarray
) -> np.ndarray:
    current = np.ones(rho.shape[1], dtype=np.float32)
    available = np.zeros(rho.shape[0], dtype=bool)
    available[eligible] = True
    selected: list[int] = []
    for _ in range(budget):
        gains = np.full(rho.shape[0], -np.inf, dtype=np.float64)
        for start in range(0, rho.shape[0], POSITION_CHUNK):
            stop = min(start + POSITION_CHUNK, rho.shape[0])
            gains[start:stop] = (1.0 - rho[start:stop]) @ current
        gains[~available] = -np.inf
        position = int(np.argmax(gains))
        selected.append(position)
        current *= rho[position]
        available[position] = False
    return np.asarray(selected, dtype=np.int32)


def static_schedule(
    rho: np.ndarray, budget: int, eligible: np.ndarray
) -> np.ndarray:
    score = np.sum(1.0 - rho, axis=1, dtype=np.float64)
    order = np.argsort(-score[eligible], kind="stable")[:budget]
    return eligible[order].astype(np.int32)


def mutual_information_schedule(
    secrets: np.ndarray,
    auth: np.ndarray,
    channels: dict[int, np.ndarray],
    budget: int,
    eligible: np.ndarray,
) -> np.ndarray:
    """Rank positions by exact single-observation mutual information."""
    gallery_size = len(secrets)
    scores = np.full(secrets.shape[1], -np.inf, dtype=np.float64)
    for bit in (0, 1):
        positions = eligible[auth[eligible] == bit]
        for start in range(0, len(positions), POSITION_CHUNK):
            chunk = positions[start : start + POSITION_CHUNK]
            rows = channels[bit][secrets[:, chunk].T]
            mixture = np.mean(rows, axis=1)
            log_rows = np.zeros_like(rows)
            log_mixture = np.zeros_like(mixture)
            np.log(rows, out=log_rows, where=rows > 0)
            np.log(mixture, out=log_mixture, where=mixture > 0)
            scores[chunk] = np.sum(
                rows * (log_rows - log_mixture[:, None, :]), axis=(1, 2)
            ) / gallery_size
    order = np.argsort(-scores[eligible], kind="stable")[:budget]
    return eligible[order].astype(np.int32)


def random_schedules(
    auth: np.ndarray, count: int, seed: int, budget: int
) -> list[np.ndarray]:
    eligible = np.flatnonzero(auth == 0)
    result: list[np.ndarray] = []
    for schedule_id in range(count):
        rng = np.random.default_rng(stable_seed(seed, "auth0", schedule_id))
        result.append(
            eligible[rng.permutation(len(eligible))[:budget]].astype(np.int32)
        )
    return result


def target_success_bounds(
    schedule: np.ndarray,
    rho: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    gallery_size: int,
) -> np.ndarray:
    pair_overlap = np.prod(rho[schedule].astype(np.float64), axis=0)
    per_target_error = np.bincount(
        pair_left, weights=pair_overlap, minlength=gallery_size
    ) + np.bincount(pair_right, weights=pair_overlap, minlength=gallery_size)
    return 1.0 - np.minimum(1.0, per_target_error)


def success_bound(
    schedule: np.ndarray,
    rho: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    gallery_size: int,
) -> float:
    return float(
        np.mean(
            target_success_bounds(
                schedule, rho, pair_left, pair_right, gallery_size
            )
        )
    )


def sample_observations(
    positions: np.ndarray,
    secrets: np.ndarray,
    auth: np.ndarray,
    cdfs: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    observations: dict[int, np.ndarray] = {}
    for position in positions:
        bit = int(auth[position])
        rows = cdfs[bit][secrets[:, position]]
        draws = rng.random(len(secrets))
        observations[int(position)] = np.argmax(
            draws[:, None] <= rows, axis=1
        ).astype(np.uint8)
    return observations


def top1_accuracy(
    schedule: np.ndarray,
    observations: dict[int, np.ndarray],
    secrets: np.ndarray,
    auth: np.ndarray,
    log_channels: dict[int, np.ndarray],
) -> float:
    gallery_size = len(secrets)
    scores = np.zeros((gallery_size, gallery_size), dtype=np.float64)
    for position in schedule:
        bit = int(auth[position])
        candidate_values = secrets[:, position]
        observed_values = observations[int(position)]
        scores += log_channels[bit][
            candidate_values[:, None], observed_values[None, :]
        ]
    predictions = np.argmax(scores, axis=0)
    return float(np.mean(predictions == np.arange(gallery_size)))


def summarize(values: np.ndarray) -> dict[str, float]:
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "ci95_low": float(np.mean(values) - 1.96 * standard_error),
        "ci95_high": float(np.mean(values) + 1.96 * standard_error),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat one-shadow linkage under fresh sharing randomness."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="BSDS500 data directory containing images/train and images/test",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--participant", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--gallery-split", choices=tuple(EXPECTED_GALLERY_SIZES), default="test"
    )
    parser.add_argument(
        "--random-schedules", type=int, default=DEFAULT_RANDOM_SCHEDULES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "linkage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    secrets, auth, auth_source = load_gallery(
        args.dataset_root.resolve(), args.gallery_split
    )
    gallery_size = len(secrets)
    eligible = np.flatnonzero(auth == 0).astype(np.int32)
    all_trial_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "gallery_prior": "uniform",
        "gallery_split": args.gallery_split,
        "gallery_size": gallery_size,
        "participant": args.participant,
        "budget": args.budget,
        "trials": args.trials,
        "random_schedules": args.random_schedules,
        "authentication_source": auth_source,
        "candidate_branch": 0,
        "candidate_positions": int(len(eligible)),
        "readings": {},
    }

    for reading, coefficient_stop in READINGS.items():
        channels = build_channel(coefficient_stop, args.participant)
        rho, pair_left, pair_right = build_position_overlaps(secrets, auth, channels)
        greedy = greedy_schedule(rho, args.budget, eligible)
        static = static_schedule(rho, args.budget, eligible)
        mi_rank = mutual_information_schedule(
            secrets, auth, channels, args.budget, eligible
        )
        random = random_schedules(
            auth, args.random_schedules, args.seed, args.budget
        )
        fixed_bounds = {
            "greedy": success_bound(
                greedy, rho, pair_left, pair_right, gallery_size
            ),
            "static": success_bound(
                static, rho, pair_left, pair_right, gallery_size
            ),
            "mi_rank": success_bound(
                mi_rank, rho, pair_left, pair_right, gallery_size
            ),
        }
        random_bounds = np.asarray(
            [
                success_bound(
                    schedule, rho, pair_left, pair_right, gallery_size
                )
                for schedule in random
            ]
        )

        log_channels: dict[int, np.ndarray] = {}
        cdfs: dict[int, np.ndarray] = {}
        for bit in (0, 1):
            log_channel = np.full(channels[bit].shape, -np.inf, dtype=np.float64)
            positive = channels[bit] > 0
            log_channel[positive] = np.log(channels[bit][positive])
            log_channels[bit] = log_channel
            cdf = np.cumsum(channels[bit], axis=1)
            cdf[:, -1] = 1.0
            cdfs[bit] = cdf

        trial_values = {
            "greedy": [],
            "static": [],
            "mi_rank": [],
            "random_auth0": [],
        }
        for trial in range(args.trials):
            random_schedule = random[trial % len(random)]
            schedules = {
                "greedy": greedy,
                "static": static,
                "mi_rank": mi_rank,
                "random_auth0": random_schedule,
            }
            union = np.unique(np.concatenate(list(schedules.values())))
            rng = np.random.default_rng(
                stable_seed(
                    args.seed,
                    f"participant{args.participant}",
                    reading,
                    "trial",
                    trial,
                    args.gallery_split,
                )
            )
            observations = sample_observations(
                union, secrets, auth, cdfs, rng
            )
            for method, schedule in schedules.items():
                value = top1_accuracy(
                    schedule, observations, secrets, auth, log_channels
                )
                trial_values[method].append(value)
                all_trial_rows.append(
                    {
                        "reading": reading,
                        "trial": trial,
                        "method": method,
                        "top1": value,
                    }
                )

        arrays = {
            method: np.asarray(values, dtype=np.float64)
            for method, values in trial_values.items()
        }
        summary["readings"][reading] = {
            "top1": {method: summarize(values) for method, values in arrays.items()},
            "paired_win_rate_greedy_over_random": float(
                np.mean(arrays["greedy"] > arrays["random_auth0"])
            ),
            "paired_nonloss_rate_greedy_vs_random": float(
                np.mean(arrays["greedy"] >= arrays["random_auth0"])
            ),
            "paired_win_rate_greedy_over_static": float(
                np.mean(arrays["greedy"] > arrays["static"])
            ),
            "paired_win_rate_greedy_over_mi_rank": float(
                np.mean(arrays["greedy"] > arrays["mi_rank"])
            ),
            "paired_top1_difference": {
                "greedy_minus_random": summarize(
                    arrays["greedy"] - arrays["random_auth0"]
                ),
                "greedy_minus_static": summarize(
                    arrays["greedy"] - arrays["static"]
                ),
                "greedy_minus_mi_rank": summarize(
                    arrays["greedy"] - arrays["mi_rank"]
                ),
            },
            "bounds": {
                **fixed_bounds,
                "random_mean": float(np.mean(random_bounds)),
                "random_p05": float(np.percentile(random_bounds, 5)),
                "random_p50": float(np.percentile(random_bounds, 50)),
                "random_p95": float(np.percentile(random_bounds, 95)),
                "random_max": float(np.max(random_bounds)),
                "fraction_random_below_greedy": float(
                    np.mean(random_bounds < fixed_bounds["greedy"])
                ),
            },
            "greedy_positions": [int(value) for value in greedy],
            "static_positions": [int(value) for value in static],
            "mi_rank_positions": [int(value) for value in mi_rank],
        }

    summary["runtime_seconds"] = time.perf_counter() - started
    split_suffix = "" if args.gallery_split == "test" else f"_{args.gallery_split}"
    write_csv(
        output_dir
        / f"repeated_shadow_trials{split_suffix}_p{args.participant}_b{args.budget}.csv",
        all_trial_rows,
    )
    (
        output_dir
        / f"repeated_shadow_summary{split_suffix}_p{args.participant}_b{args.budget}.json"
    ).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
