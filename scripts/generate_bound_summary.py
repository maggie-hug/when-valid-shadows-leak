from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_repeated_linkage import (
    DEFAULT_RANDOM_SCHEDULES,
    DEFAULT_SEED,
    READINGS,
    build_channel,
    build_position_overlaps,
    greedy_schedule,
    load_gallery,
    mutual_information_schedule,
    random_schedules,
    static_schedule,
    success_bound,
    target_success_bounds,
)


BUDGETS = (1, 2, 4, 8, 12, 16, 24, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact certificate summaries for the paper's results figure."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "bound_summary.csv",
    )
    parser.add_argument(
        "--random-schedules", type=int, default=DEFAULT_RANDOM_SCHEDULES
    )
    parser.add_argument(
        "--branch-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "branch_ablation.csv",
    )
    parser.add_argument(
        "--target-output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "target_certificate_values.csv",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secrets, auth, _ = load_gallery(args.dataset_root.resolve(), "test")
    gallery_files = [
        path.name
        for path in sorted(
            (args.dataset_root.resolve() / "images" / "test").glob("*.jpg")
        )
    ]
    eligible = np.flatnonzero(auth == 0).astype(np.int32)
    gallery_size = len(secrets)
    if len(gallery_files) != gallery_size:
        raise RuntimeError("gallery filename count does not match the loaded gallery")
    rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []

    for participant in (1, 2):
        for reading, coefficient_stop in READINGS.items():
            channels = build_channel(coefficient_stop, participant)
            rho, pair_left, pair_right = build_position_overlaps(
                secrets, auth, channels
            )
            max_budget = max(BUDGETS)
            schedules = {
                "static": static_schedule(rho, max_budget, eligible),
                "mi_rank": mutual_information_schedule(
                    secrets, auth, channels, max_budget, eligible
                ),
                "greedy": greedy_schedule(rho, max_budget, eligible),
            }
            candidate_pools = {
                "auth0": eligible,
                "auth1": np.flatnonzero(auth == 1).astype(np.int32),
                "all": np.arange(len(auth), dtype=np.int32),
            }
            branch_schedules = {
                name: greedy_schedule(rho, 12, positions)
                for name, positions in candidate_pools.items()
            }
            random = random_schedules(
                auth, args.random_schedules, args.seed, max_budget
            )

            for budget in BUDGETS:
                for method, schedule in schedules.items():
                    bound = success_bound(
                        schedule[:budget],
                        rho,
                        pair_left,
                        pair_right,
                        gallery_size,
                    )
                    rows.append(
                        {
                            "participant": participant,
                            "reading": reading,
                            "budget": budget,
                            "method": method,
                            "bound": bound,
                            "p05": "",
                            "p50": "",
                            "p95": "",
                            "maximum": "",
                        }
                    )

                random_bounds = np.asarray(
                    [
                        success_bound(
                            schedule[:budget],
                            rho,
                            pair_left,
                            pair_right,
                            gallery_size,
                        )
                        for schedule in random
                    ],
                    dtype=np.float64,
                )
                rows.append(
                    {
                        "participant": participant,
                        "reading": reading,
                        "budget": budget,
                        "method": "random_auth0",
                        "bound": float(np.mean(random_bounds)),
                        "p05": float(np.percentile(random_bounds, 5)),
                        "p50": float(np.percentile(random_bounds, 50)),
                        "p95": float(np.percentile(random_bounds, 95)),
                        "maximum": float(np.max(random_bounds)),
                    }
                )

            for pool_name, schedule in branch_schedules.items():
                for budget in (8, 12):
                    branch_rows.append(
                        {
                            "participant": participant,
                            "reading": reading,
                            "candidate_pool": pool_name,
                            "candidate_positions": len(candidate_pools[pool_name]),
                            "budget": budget,
                            "bound": success_bound(
                                schedule[:budget],
                                rho,
                                pair_left,
                                pair_right,
                                gallery_size,
                            ),
                            "same_schedule_as_auth0": bool(
                                np.array_equal(
                                    schedule[:budget],
                                    branch_schedules["auth0"][:budget],
                                )
                            ),
                        }
                    )

            target_bounds = target_success_bounds(
                branch_schedules["auth0"],
                rho,
                pair_left,
                pair_right,
                gallery_size,
            )
            for target_index, bound in enumerate(target_bounds):
                target_rows.append(
                    {
                        "participant": participant,
                        "reading": reading,
                        "budget": 12,
                        "candidate_pool": "auth0",
                        "target_index": target_index,
                        "target_file": gallery_files[target_index],
                        "bound": float(bound),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.branch_output.parent.mkdir(parents=True, exist_ok=True)
    with args.branch_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(branch_rows[0]))
        writer.writeheader()
        writer.writerows(branch_rows)
    args.target_output.parent.mkdir(parents=True, exist_ok=True)
    with args.target_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"wrote {len(branch_rows)} rows to {args.branch_output}")
    print(f"wrote {len(target_rows)} rows to {args.target_output}")


if __name__ == "__main__":
    main()
