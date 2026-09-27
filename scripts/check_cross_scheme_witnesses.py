#!/usr/bin/env python3
"""Independent, standard-library-only finite checks; never writes files.

Default output includes complete accepted sets and exact TCSVT21 Fraction
pmfs. --summary omits long sets; --csv prints the compact evidence table.

Scope: paper-text coefficient-stage interpretations, not JPEG end-to-end
replication. TCSVT21 fixes public authentication bit zero and samples its
hidden 00/11 patterns before uniform coefficient rejection. MBE22 checks
reachable sets without assigning probabilities to the source solver.
SBC24 preserves regulation and explicitly uses odd-position metadata for
authorized recovery. Its stability check includes dequantization with a
common grayscale JPEG QF=90 table and checks exact pre-rounding spatial
values for the declared DC-only blocks. Eight- and nine-bit prefix readings are kept
separate; neither reading is claimed to be the source's unique semantics.

Source locators: TCSVT21 Algorithm 1; MBE22 Eq. (2.7), Algorithm 1 Steps
4--7; SBC24 Algorithm 1 and Eqs. (13)--(20). The SBC24 nine-bit witness is
a new finite verification for this revision, not an original image result.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from fractions import Fraction


P = 257
# Luminance table in natural (row-major) order, using the libjpeg/Pillow
# quality-90 convention. Input images, output shares, and the contemplated
# recompression use this same table; QF alone is not a JPEG file parameter.
JPEG_QUALITY = 90
JPEG_LUMA_QF90 = (
    3, 2, 2, 3, 5, 8, 10, 12,
    2, 2, 3, 4, 5, 12, 12, 11,
    3, 3, 3, 5, 8, 11, 14, 11,
    3, 3, 4, 6, 10, 17, 16, 12,
    4, 4, 7, 11, 14, 22, 21, 15,
    5, 7, 11, 13, 16, 21, 23, 18,
    10, 13, 16, 17, 21, 24, 24, 20,
    14, 18, 19, 20, 22, 20, 21, 20,
)
CSV_FIELDS = (
    "scheme", "reading", "p", "k", "n", "evaluation_points", "condition",
    "observed_participant", "secret0", "secret1", "accepted_counts0",
    "accepted_counts1", "reachable_count0", "reachable_count1", "tv",
    "probability_semantics", "recovery_scope",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_quantization_tables(sharing: tuple[int, ...],
                              recompression: tuple[int, ...]) -> None:
    require(sharing == recompression, "SBC24 requires QM1 = QM2")
    require(len(sharing) == 64, "SBC24 requires an 8-by-8 quantization table")
    require(all(isinstance(step, int) and 2 <= step <= 255 for step in sharing),
            "SBC24 requires every baseline JPEG quantization step in [2,255]")


def dc_only_spatial(dc: int, quantization: tuple[int, ...]) -> Fraction:
    # Entropy decoding yields a quantized coefficient: dequantize before IDCT.
    return Fraction(dc * quantization[0], 8)


def in_stability_interval(value: Fraction) -> bool:
    return Fraction(-128) <= value < 127


def check_quantization_regressions() -> None:
    check_quantization_tables(JPEG_LUMA_QF90, JPEG_LUMA_QF90)
    require(JPEG_LUMA_QF90[0] == 3, "SBC24 QF=90 DC quantization step changed")
    # A quantized DC of 384 would incorrectly pass if dequantization were
    # omitted: 384/8=48, whereas its actual spatial value is 144 and overflows.
    require(in_stability_interval(Fraction(384, 8)), "Invalid overflow control")
    require(not in_stability_interval(dc_only_spatial(384, JPEG_LUMA_QF90)),
            "SBC24 overflow control failed: dequantization may be missing")
    unit_step = (1,) + JPEG_LUMA_QF90[1:]
    mismatched = (4,) + JPEG_LUMA_QF90[1:]
    for sharing, recompression in ((unit_step, unit_step),
                                   (JPEG_LUMA_QF90, mismatched)):
        try:
            check_quantization_tables(sharing, recompression)
        except AssertionError:
            continue
        raise AssertionError("SBC24 accepted invalid quantization side conditions")


def raw_shares(secret: int, coefficient: int) -> tuple[int, int]:
    return (secret + coefficient) % P, (secret + 2 * coefficient) % P


def recover(raw: tuple[int, int]) -> int:
    return (2 * raw[0] - raw[1]) % P


def check_recovery(secret: int, coefficients: list[int]) -> None:
    require(bool(coefficients), f"Empty accepted set for secret {secret}")
    for coefficient in coefficients:
        require(recover(raw_shares(secret, coefficient)) == secret,
                "Raw polynomial interpolation failed")


def next_prime(value: int) -> int:
    candidate = value + 1
    while True:
        if candidate >= 2 and all(candidate % d for d in range(2, candidate)
                                  if d * d <= candidate):
            return candidate
        candidate += 1


def prefix(value: int, width: int) -> int:
    return value // (2 ** (width - 4))


def prefix_candidates(secret: int, cover: tuple[int, int], width: int) -> list[int]:
    return [a for a in range(P)
            if all((width != 8 or value < 256)
                   and prefix(value, width) == prefix(center, width)
                   for value, center in zip(raw_shares(secret, a), cover))]


def exact_tv(left: dict[int, Fraction], right: dict[int, Fraction]) -> Fraction:
    return sum((abs(left.get(y, Fraction()) - right.get(y, Fraction()))
                for y in left.keys() | right.keys()), Fraction()) / 2


def summary_row(scheme: str, reading: str, condition: str, participant: int,
                secrets: tuple[int, int], counts: list[str], views: list[set[int]],
                probability_semantics: str, recovery_scope: str) -> dict[str, object]:
    require(not views[0].intersection(views[1]), f"{scheme}: supports overlap")
    require(all(views), f"{scheme}: empty reachable output set")
    return dict(zip(CSV_FIELDS, (
        scheme, reading, P, 2, 2, "1;2", condition, participant,
        secrets[0], secrets[1], counts[0], counts[1], len(views[0]),
        len(views[1]), "1", probability_semantics, recovery_scope,
    )))


def tcsvt_witness(stop: int, participant: int,
                  secrets: tuple[int, int]) -> dict[str, object]:
    reading = "full-field" if stop == 257 else "byte-domain"
    distributions: list[dict[int, Fraction]] = []
    details: list[dict[str, object]] = []
    counts: list[str] = []
    for secret in secrets:
        pmf: dict[int, Fraction] = {}
        accepted: dict[str, list[int]] = {}
        for bit in (0, 1):
            pattern = str(bit) * 2
            candidates = [a for a in range(stop)
                          if all(value < 256 and (value & 15).bit_count() % 2 == bit
                                 for value in raw_shares(secret, a))]
            check_recovery(secret, candidates)
            accepted[pattern] = candidates
            # Hidden pattern has mass 1/2; only the coefficient is resampled.
            mass = Fraction(1, 2 * len(candidates))
            for coefficient in candidates:
                y = raw_shares(secret, coefficient)[participant - 1]
                pmf[y] = pmf.get(y, Fraction()) + mass
        require(sum(pmf.values(), Fraction()) == 1, "TCSVT21 pmf not normalized")
        distributions.append(pmf)
        counts.append(";".join(f"{g}:{len(a)}" for g, a in accepted.items()))
        details.append({"secret": secret, "accepted_by_pattern": accepted,
                        "pmf": {str(y): str(pmf[y]) for y in sorted(pmf)}})
    require(exact_tv(*distributions) == 1, "TCSVT21 exact TV is not one")
    row = summary_row("TCSVT21", reading, "authentication_bit=0", participant,
                      secrets, counts, [set(pmf) for pmf in distributions],
                      "half_per_pattern_uniform_coefficient_rejection",
                      "raw_shared_symbol")
    return {"summary": row, "hidden_pattern_masses": {"00": "1/2", "11": "1/2"},
            "details": details, "exact_tv": str(exact_tv(*distributions))}


def mbe_witness(width: int, secrets: tuple[int, int]) -> dict[str, object]:
    expected = {8: ([15], list(range(8))), 9: ([31], list(range(16)))}[width]
    details: list[dict[str, object]] = []
    views: list[set[int]] = []
    counts: list[str] = []
    for index, secret in enumerate(secrets):
        # Complete 16-by-16 JPEG inputs: four DC-only 8-by-8 blocks in
        # row-major order. Triples contain (secret, cover1, cover2); every
        # AC coefficient is zero, and num=4 selects DC plus three AC terms.
        num = 4
        dc_triples = ((secret, 128, 128), (0, 0, 0),
                      (252, 252, 252), (0, 0, 0))
        coefficient_blocks = [(dc,) + ((0, 0, 0),) * 63 for dc in dc_triples]
        selected_triples = [triple for block in coefficient_blocks
                            for triple in block[:num]]
        selected = [v for triple in selected_triples for v in triple]
        shift = max(0, -min(selected))
        modulus = next_prime(max(v + shift for v in selected))
        require((shift, modulus) == (0, P), "MBE22 preprocessing anchors failed")
        for position, (value, cover1, cover2) in enumerate(selected_triples[1:], 1):
            require(0 in prefix_candidates(value, (cover1, cover2), width),
                    f"MBE22 non-target selected coefficient {position} has no z=0 solution")
        candidates = prefix_candidates(secret, (128, 128), width)
        require(candidates == expected[index], "MBE22 accepted set changed")
        check_recovery(secret, candidates)
        reachable = {raw_shares(secret, a)[0] for a in candidates}
        views.append(reachable)
        counts.append(str(len(candidates)))
        details.append({"secret": secret, "accepted": candidates,
                        "reachable_view": sorted(reachable),
                        "preprocessing_shift": shift, "preprocessing_prime": modulus,
                        "anchor_a0_feasible": True})
    row = summary_row("MBE22", f"w={width}", "cover=128;128;delta=4", 1,
                      secrets, counts, views, "any_solver_returning_an_accepted_value",
                      "selected_coefficients_only")
    return {"summary": row, "details": details,
            "tv_justification": "disjoint reachable sets; solver probabilities unspecified"}


def sbc_witness(width: int, secrets: tuple[int, int]) -> dict[str, object]:
    expected = {8: (32, 31), 9: (64, 2)}[width]
    expected_spatial = {
        8: ((Fraction(45), Fraction(3)), (Fraction(45), Fraction(21, 8))),
        9: ((Fraction(42), Fraction(6)), (Fraction(48), Fraction(0))),
    }[width]
    sharing_qm = JPEG_LUMA_QF90
    recompression_qm = JPEG_LUMA_QF90
    check_quantization_tables(sharing_qm, recompression_qm)
    details: list[dict[str, object]] = []
    views: list[set[int]] = []
    for index, secret in enumerate(secrets):
        selected = (secret, 252, 0)
        shift, modulus = abs(min(selected)), next_prime(max(selected))
        # Source Eq. (15): floor((k-1)/n * log2(p-1)); here p-1=256.
        delta = ((modulus - 1).bit_length() - 1) // 2
        require((shift, modulus, delta) == (0, P, 4), "SBC24 parameter derivation failed")
        candidates = prefix_candidates(secret, (252, 0), width)
        require(candidates == [expected[index]], "SBC24 is not the expected singleton")
        check_recovery(secret, candidates)
        raw = raw_shares(secret, candidates[0])
        regulated = tuple((u + 1) // 2 for u in raw)
        odd = tuple(u % 2 for u in raw)
        restored = tuple(2 * y - bit for y, bit in zip(regulated, odd))
        require(restored == raw, "SBC24 odd metadata failed to undo regulation")
        require(recover(restored) == secret, "SBC24 authorized recovery failed")
        # With l=1 and zero AC tails, the dequantized IDCT is d*Q_DC/8.
        # Check the exact spatial values before rounding or clipping; these
        # witnesses lie safely inside the source Eqs. (10)/(19) interval.
        spatial = tuple(dc_only_spatial(dc, sharing_qm) for dc in regulated)
        require(spatial == expected_spatial[index], "SBC24 dequantized values changed")
        require(all(in_stability_interval(value) for value in spatial),
                "SBC24 DC-only pre-rounding stability failed")
        views.append({regulated[1]})
        details.append({"secret": secret, "accepted": candidates,
                        "raw_shares": list(raw), "regulated_coefficients": list(regulated),
                        "odd_bits": list(odd), "restored_raw_shares": list(restored),
                        "recovered_secret": recover(restored),
                        "constant_idct_values": [str(v) for v in spatial],
                        "stability_checked_before_rounding_or_clipping": True,
                        "stable": True,
                        "preprocessing_shift": shift, "preprocessing_prime": modulus})
    pmfs = [{next(iter(view)): Fraction(1)} for view in views]
    require(exact_tv(*pmfs) == 1, "SBC24 regulated-coefficient TV is not one")
    row = summary_row("SBC24", f"w={width}",
                      "cover=252;0;delta=4;l=1;alpha=1/2;QF=90;Q_DC=3;QM1=QM2;min_QM=2", 2,
                      secrets, ["1", "1"], views, "singleton_candidates",
                      "DC_only_block_with_odd_metadata")
    return {"summary": row, "details": details, "exact_tv": "1",
            "quantization": {"quality_factor": JPEG_QUALITY,
                             "table_order": "natural_row_major",
                             "sharing_table": list(sharing_qm),
                             "recompression_table": list(recompression_qm),
                             "dc_step": sharing_qm[0],
                             "minimum_step": min(sharing_qm)},
            "provenance": "new_revision_check" if width == 9 else "existing_USENIX_witness"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true", help="print compact JSON")
    mode.add_argument("--csv", action="store_true", help="print compact CSV to stdout")
    args = parser.parse_args()
    check_quantization_regressions()
    results = [tcsvt_witness(stop, participant, secrets)
               for stop in (257, 256)
               for participant, secrets in ((1, (3, 252)), (2, (0, 16)))]
    results.extend(mbe_witness(width, secrets)
                   for width, secrets in ((8, (113, 129)), (9, (97, 129))))
    results.extend(sbc_witness(width, secrets)
                   for width, secrets in ((8, (208, 209)), (9, (160, 253))))
    summaries = [result["summary"] for result in results]
    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    else:
        print(json.dumps({"status": "all_passed", "witness_count": len(results),
                          "cases": summaries if args.summary else results}, indent=2))


if __name__ == "__main__":
    main()
