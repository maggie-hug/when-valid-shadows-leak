"""Validate a source-rule generator against a separately implemented channel.

The reference module is used for the oracle and decoder ONLY. The generator
receives secret pixels, public bits, coefficient domain, and an RNG, never W.
All output files belong to this validation directory, not manuscript data/.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import scipy
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
sys.path.insert(0, str(PAPER / "scripts"))
import run_repeated_linkage as reference
import check_tcsvt21_trace as trace_reference
import generator


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stream(config: dict, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([config["base_seed"], *parts]))


def channel_pair(stop: int) -> dict[int, dict[int, np.ndarray]]:
    return {participant: reference.build_channel(stop, participant) for participant in (1, 2)}


def check_generated(secret: np.ndarray, result: dict) -> dict:
    first = result["shadow1"].astype(np.int64)
    second = result["shadow2"].astype(np.int64)
    coefficient = result["coefficient"].astype(np.int64)
    checks = {
        "reconstruction_errors": int(np.count_nonzero((2 * first - second) % 257 != secret)),
        "field_evaluation_errors": int(np.count_nonzero(
            ((secret.astype(np.int64) + coefficient) % 257 != first)
            | ((secret.astype(np.int64) + 2 * coefficient) % 257 != second))),
        "parity_errors": int(np.count_nonzero(
            (generator.source_parity(first) != result["pattern1"])
            | (generator.source_parity(second) != result["pattern2"]))),
        "pixels": int(secret.size),
        "coefficient_draws": int(result["attempts"].sum()),
        "max_attempts": int(result["attempts"].max()),
    }
    if any(checks[key] for key in ("reconstruction_errors", "field_evaluation_errors", "parity_errors")):
        raise AssertionError(checks)
    if result["shadow1"].dtype != np.uint8 or result["shadow2"].dtype != np.uint8:
        raise AssertionError("Released observations must be byte arrays")
    return checks


def exact_stage(config: dict, out: Path) -> dict:
    rows = []
    rng_pattern_counts = {}
    for bit in (0, 1):
        counter = Counter()
        for coin in (0, 1):
            for permutation in range(6):
                first, second, _ = generator.hidden_from_primitives(bit, coin, permutation)
                counter[(int(first), int(second))] += 1
        actual = {pair: count / 12 for pair, count in counter.items()}
        expected = {pattern: float(mass) for pattern, mass in trace_reference.PATTERN_LAWS[bit]}
        if actual != expected:
            raise AssertionError((bit, actual, expected))
        rng_pattern_counts[str(bit)] = {str(pair): count for pair, count in counter.items()}
    max_channel_difference = 0.0
    decisions = 0
    groups = 0
    for reading, stop in config["coefficient_stops"].items():
        channels = channel_pair(stop)
        coefficients = np.arange(stop, dtype=np.int64)
        compiled = {p: {b: np.zeros((256, 256)) for b in (0, 1)} for p in (1, 2)}
        for secret in range(256):
            outputs = generator.source_shares(secret, coefficients)
            for g1, g2 in itertools.product((0, 1), repeat=2):
                mask = generator.source_accepts(secret, coefficients, g1, g2)
                accepted = coefficients[mask]
                expected = trace_reference.accepted_coefficients(secret, (g1, g2), range(stop))
                if tuple(accepted) != expected:
                    raise AssertionError((reading, secret, g1, g2, "accepted-set mismatch"))
                if not len(accepted):
                    raise AssertionError("Nonterminating state")
                groups += 1
                decisions += stop
                for bit in (0, 1):
                    primitive_count = rng_pattern_counts[str(bit)].get(str((g1, g2)), 0)
                    if not primitive_count:
                        continue
                    for participant in (1, 2):
                        values = outputs[participant - 1][mask].astype(int)
                        compiled[participant][bit][secret] += (
                            np.bincount(values, minlength=256) * (primitive_count / 12 / len(accepted)))
        for participant in (1, 2):
            for bit in (0, 1):
                difference = float(np.max(np.abs(compiled[participant][bit] - channels[participant][bit])))
                max_channel_difference = max(max_channel_difference, difference)
                if difference > config["exact_channel_max_abs_tolerance"]:
                    raise AssertionError((reading, participant, bit, difference))
                rows.append({"reading": reading, "participant": participant, "public_bit": bit,
                             "channel_rows": 256, "max_abs_channel_difference": difference})
    result = {"status": "completed", "primitive_pattern_counts": rng_pattern_counts,
              "accepted_groups_checked": groups, "coefficient_state_decisions": decisions,
              "max_abs_channel_difference": max_channel_difference, "rows": rows}
    write_json(out / "exact_checks.json", result)
    return result


def gof(observed: np.ndarray, probability: np.ndarray) -> dict:
    total = int(observed.sum())
    positive = probability > 0
    impossible = int(observed[~positive].sum())
    expected = total * probability[positive]
    statistic = float(np.sum((observed[positive] - expected) ** 2 / expected))
    dof = int(positive.sum() - 1)
    p_value = 0.0 if impossible else float(chi2.sf(statistic, dof))
    return {"draws": total, "impossible_observations": impossible,
            "chi_square": statistic, "degrees_freedom": dof, "p_value": p_value,
            "min_expected_count": float(expected.min()),
            "empirical_tv": float(0.5 * np.abs(observed / total - probability).sum())}


def holm(rows: list[dict], alpha: float) -> int:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    previous = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, max(previous, (len(rows) - rank) * rows[index]["p_value"]))
        previous = adjusted
        rows[index]["holm_adjusted_p"] = adjusted
        rows[index]["reject_at_familywise_alpha"] = adjusted < alpha
    return sum(row["reject_at_familywise_alpha"] for row in rows)


def marginal_stage(config: dict, out: Path) -> dict:
    rows = []
    controls = []
    arrays = {}
    all_checks = []
    n = config["row_draws"]
    chunk = config["row_chunk_secrets"]
    for domain_index, (reading, stop) in enumerate(config["coefficient_stops"].items()):
        channels = channel_pair(stop)
        for bit in (0, 1):
            counts = {p: np.zeros((256, 256), dtype=np.int64) for p in (1, 2)}
            patterns = np.zeros((256, 4), dtype=np.int64)
            for start in range(0, 256, chunk):
                secrets = np.broadcast_to(np.arange(start, min(256, start + chunk), dtype=np.uint8)[:, None],
                                          (min(chunk, 256 - start), n)).copy()
                generated = generator.generate(secrets, np.uint8(bit), stop,
                                               stream(config, 1, domain_index, bit, start))
                all_checks.append(check_generated(secrets, generated))
                for offset, secret in enumerate(range(start, start + len(secrets))):
                    for participant in (1, 2):
                        counts[participant][secret] = np.bincount(generated[f"shadow{participant}"][offset], minlength=256)
                    patterns[secret] = np.bincount(
                        generated["pattern1"][offset] * 2 + generated["pattern2"][offset], minlength=4)
            pattern_probability = np.zeros(4)
            for pattern, weight in trace_reference.PATTERN_LAWS[bit]:
                pattern_probability[pattern[0] * 2 + pattern[1]] = float(weight)
            arrays[f"{reading}_b{bit}_patterns"] = patterns
            for secret in range(256):
                rows.append({"reading": reading, "public_bit": bit, "secret": secret,
                             "observation": "hidden_pattern", **gof(patterns[secret], pattern_probability)})
                for participant in (1, 2):
                    rows.append({"reading": reading, "public_bit": bit, "secret": secret,
                                 "observation": f"participant{participant}",
                                 **gof(counts[participant][secret], channels[participant][bit][secret])})
            for participant in (1, 2):
                arrays[f"{reading}_b{bit}_p{participant}"] = counts[participant]
            print(f"MARGINAL {reading} bit={bit} complete", flush=True)
            control_values = np.asarray(config["negative_control_secrets"], dtype=np.uint8)
            secrets = np.broadcast_to(control_values[:, None], (len(control_values), n)).copy()
            generated = generator.generate(secrets, np.uint8(bit), stop,
                                           stream(config, 2, domain_index, bit),
                                           negative_control_resample_pattern=True)
            check_generated(secrets, generated)
            for participant in (1, 2):
                control_counts = np.stack([np.bincount(v, minlength=256) for v in generated[f"shadow{participant}"]])
                arrays[f"negative_{reading}_b{bit}_p{participant}"] = control_counts
                for index, secret in enumerate(control_values):
                    controls.append({"reading": reading, "public_bit": bit, "secret": int(secret),
                                     "observation": f"participant{participant}",
                                     **gof(control_counts[index], channels[participant][bit][secret])})
    rejected = holm(rows, config["familywise_alpha"])
    control_rejected = holm(controls, config["familywise_alpha"])
    write_csv(out / "marginal_gof.csv", rows)
    write_csv(out / "negative_control_gof.csv", controls)
    np.savez_compressed(out / "marginal_counts.npz", **arrays)
    participant_rows = [row for row in rows if row["observation"] != "hidden_pattern"]
    result = {"status": "completed", "draws_per_secret_bit_domain": n,
              "source_pixel_pairs": sum(row["pixels"] for row in all_checks),
              "coefficient_draws": sum(row["coefficient_draws"] for row in all_checks),
              "max_attempts": max(row["max_attempts"] for row in all_checks),
              "gof_tests": len(rows), "participant_row_tests": len(participant_rows),
              "familywise_alpha": config["familywise_alpha"], "holm_rejections": rejected,
              "impossible_observations": sum(row["impossible_observations"] for row in rows),
              "minimum_expected_cell_count": min(row["min_expected_count"] for row in rows),
              "mean_empirical_participant_tv": float(np.mean([row["empirical_tv"] for row in participant_rows])),
              "max_empirical_participant_tv": max(row["empirical_tv"] for row in participant_rows),
              "negative_control_tests": len(controls), "negative_control_holm_rejections": control_rejected,
              "negative_control_is_deliberately_wrong": True,
              "interpretation": "No corrected rejection is non-detection, not a proof of distributional equality"}
    write_json(out / "marginal_summary.json", result)
    return result


def load_images(config: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = Path(config["dataset_root"])
    paths = sorted((root / "images" / config["gallery_split"]).glob("*.jpg"))
    if len(paths) != config["gallery_size"]:
        raise AssertionError(("gallery size", len(paths)))
    def convert(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("L")
            side = min(image.size)
            x = (image.width - side) // 2
            y = (image.height - side) // 2
            return np.array(image.crop((x, y, x + side, y + side)).resize(
                (config["image_size"], config["image_size"]), Image.Resampling.LANCZOS), dtype=np.uint8)
    images = np.stack([convert(path) for path in paths])
    authentication = convert(root / "images" / "train" / config["authentication_source"])
    mask = (authentication > np.median(authentication)).astype(np.uint8)
    existing_images, existing_mask, existing_name = reference.load_gallery(root, config["gallery_split"])
    if not (np.array_equal(images.reshape(len(paths), -1), existing_images)
            and np.array_equal(mask.ravel(), existing_mask)
            and existing_name == config["authentication_source"]):
        raise AssertionError("Preprocessing differs from manuscript protocol")
    return images, mask, [path.name for path in paths]


def png_round_trip(shadows: np.ndarray) -> np.ndarray:
    observed = np.empty_like(shadows)
    for index, shadow in enumerate(shadows):
        buffer = io.BytesIO()
        Image.fromarray(shadow).save(buffer, format="PNG")
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            observed[index] = np.asarray(decoded, dtype=np.uint8)
    if not np.array_equal(observed, shadows):
        raise AssertionError("PNG changes a released observation")
    return observed


def predict(gallery: np.ndarray, observed: np.ndarray, positions: np.ndarray,
            mask: np.ndarray, channels: dict[int, np.ndarray]) -> np.ndarray:
    gallery = gallery.reshape(len(gallery), -1)
    observed = observed.reshape(len(observed), -1)
    scores = np.zeros((len(observed), len(gallery)), dtype=np.float64)
    logs = {}
    for bit in (0, 1):
        logs[bit] = np.full_like(channels[bit], -np.inf)
        nonzero = channels[bit] > 0
        logs[bit][nonzero] = np.log(channels[bit][nonzero])
    for position in positions:
        bit = int(mask.ravel()[position])
        scores += logs[bit][gallery[None, :, position], observed[:, None, position]]
    if not np.isfinite(scores[np.arange(len(gallery)), np.arange(len(gallery))]).all():
        raise AssertionError("A generated target has zero model likelihood")
    return np.argmax(scores, axis=1)


def statistics(values: np.ndarray) -> dict:
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    mean = float(np.mean(values))
    return {"trials": len(values), "mean_fraction": mean, "sd_fraction": float(np.std(values, ddof=1)),
            "se_fraction": standard_error, "ci95_low_fraction": mean - 1.96 * standard_error,
            "ci95_high_fraction": mean + 1.96 * standard_error}


def compare(source: np.ndarray, baseline: np.ndarray, margin: float) -> dict:
    a, b = statistics(source), statistics(baseline)
    difference = a["mean_fraction"] - b["mean_fraction"]
    error = 1.96 * math.sqrt(a["se_fraction"] ** 2 + b["se_fraction"] ** 2)
    return {"source": a, "channel_baseline": b, "difference_fraction": difference,
            "difference_ci95_low_fraction": difference - error,
            "difference_ci95_high_fraction": difference + error,
            "equivalence_margin_fraction": margin,
            "within_prespecified_margin": difference - error > -margin and difference + error < margin}


def load_baseline(participant: int, reading: str) -> np.ndarray:
    path = PAPER / "data" / f"repeated_shadow_trials_p{participant}_b12.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["reading"] == reading and row["method"] == "greedy"]
    rows.sort(key=lambda row: int(row["trial"]))
    if [int(row["trial"]) for row in rows] != list(range(1000)):
        raise AssertionError("Expected the original 1000 baseline trials")
    return np.array([float(row["top1"]) for row in rows])


def linkage_stage(config: dict, out: Path) -> dict:
    images, mask, names = load_images(config)
    trials = config["full_gallery_trials_per_domain"]
    all_rows = []
    all_source = []
    all_baseline = []
    strata = []
    total_draws = total_pixels = max_attempts = 0
    started = time.perf_counter()
    for domain_index, (reading, stop) in enumerate(config["coefficient_stops"].items()):
        channels = channel_pair(stop)
        schedules = {}
        for participant in (1, 2):
            path = PAPER / "data" / f"repeated_shadow_summary_p{participant}_b12.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if not (metadata["gallery_size"] == len(images) and metadata["budget"] == config["budget"]
                    and metadata["authentication_source"] == config["authentication_source"]):
                raise AssertionError("Frozen schedule metadata mismatch")
            schedules[participant] = np.asarray(metadata["readings"][reading]["greedy_positions"], dtype=int)
            if len(schedules[participant]) != config["budget"] or np.any(mask.ravel()[schedules[participant]]):
                raise AssertionError("Frozen positions are not 12 authentication-zero locations")
        values = {participant: [] for participant in (1, 2)}
        for trial in range(trials):
            generated = generator.generate(images, mask, stop, stream(config, 3, domain_index, trial))
            checked = check_generated(images, generated)
            total_pixels += checked["pixels"]
            total_draws += checked["coefficient_draws"]
            max_attempts = max(max_attempts, checked["max_attempts"])
            decoded = {p: png_round_trip(generated[f"shadow{p}"]) for p in (1, 2)}
            if not np.array_equal((2 * decoded[1].astype(np.int64) - decoded[2]) % 257, images):
                raise AssertionError("Reconstruction after PNG failed")
            for participant in (1, 2):
                predictions = predict(images, decoded[participant], schedules[participant], mask, channels[participant])
                correct = int(np.count_nonzero(predictions == np.arange(len(images))))
                values[participant].append(correct / len(images))
                all_rows.append({"reading": reading, "participant": participant, "trial": trial,
                                 "correct": correct, "targets": len(images), "top1": correct / len(images)})
            if trial == 0:
                np.savez_compressed(out / f"full_shadow_example_{reading}.npz", secrets=images,
                                    public_mask=mask, filenames=np.asarray(names), **generated)
                for participant in (1, 2):
                    Image.fromarray(decoded[participant][0]).save(out / f"example_{reading}_p{participant}.png")
            if (trial + 1) % 10 == 0:
                print(f"LINKAGE {reading} {trial+1}/{trials}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
        for participant in (1, 2):
            source = np.asarray(values[participant])
            baseline = load_baseline(participant, reading)
            all_source.append(source)
            all_baseline.append(baseline)
            strata.append({"reading": reading, "participant": participant,
                           "positions": schedules[participant].tolist(),
                           **compare(source, baseline, config["equivalence_margin_fraction"])})
    write_csv(out / "full_shadow_trials.csv", all_rows)
    source_aggregate = np.mean(all_source, axis=0)
    baseline_aggregate = np.mean(all_baseline, axis=0)
    aggregate = compare(source_aggregate, baseline_aggregate, config["equivalence_margin_fraction"])
    result = {"status": "completed", "gallery_filenames": names,
              "public_mask_zero_positions": int(np.count_nonzero(mask == 0)),
              "trials_per_domain": trials, "full_image_pairs": trials * len(images) * 2,
              "png_roundtrips": trials * len(images) * 4,
              "generated_pixel_pairs": total_pixels, "coefficient_draws": total_draws,
              "max_attempts": max_attempts, "reconstruction_errors": 0, "png_errors": 0,
              "strata": strata, "equal_stratum_aggregate": aggregate,
              "all_equivalence_checks_pass": all(row["within_prespecified_margin"] for row in strata) and aggregate["within_prespecified_margin"],
              "runtime_seconds": time.perf_counter() - started,
              "scope": config["scope"], "confidence_interval_scope": "Monte Carlo variation conditional on fixed gallery/mask/schedules and declared probability laws"}
    write_json(out / "linkage_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "exact", "marginal", "linkage"), default="all")
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--dataset-root", type=Path, help="BSDS500 data directory; required for linkage/all")
    parser.add_argument("--output-dir", type=Path, default=PAPER / "outputs" / "tcsvt21_generator")
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.dataset_root is not None:
        config["dataset_root"] = str(args.dataset_root.resolve())
    if args.stage in ("all", "linkage") and args.dataset_root is None:
        parser.error("--dataset-root is required for linkage/all; numerical protocol parameters remain fixed")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = ("exact", "marginal", "linkage") if args.stage == "all" else (args.stage,)
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "protocol": config, "command": sys.argv, "python": sys.version,
                "numpy": np.__version__, "pillow": PIL.__version__, "platform": platform.platform(),
                "scipy": scipy.__version__,
                "stages_requested": stages, "stages_completed": []}
    write_json(out / "run_manifest.json", manifest)
    started = time.perf_counter()
    try:
        for stage in stages:
            print(f"START {stage}", flush=True)
            result = {"exact": exact_stage, "marginal": marginal_stage, "linkage": linkage_stage}[stage](config, out)
            manifest["stages_completed"].append(stage)
            write_json(out / "run_manifest.json", manifest)
            print(f"DONE {stage}", flush=True)
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["runtime_seconds"] = time.perf_counter() - started
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
