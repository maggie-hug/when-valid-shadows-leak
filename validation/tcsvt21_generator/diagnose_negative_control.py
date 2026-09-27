"""Post-hoc deterministic diagnosis; does not rerun or alter frozen statistics."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import generator
from validate_generator import channel_pair, trace_reference, write_json


class ScriptedDraws(np.random.Generator):
    """A deterministic execution trace, not a source of experimental randomness."""
    def __init__(self, stop: int):
        super().__init__(np.random.PCG64(0))
        self.stop = stop
        self.coin_calls = 0
        self.coefficient_calls = 0
        self.events = []

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if low != 0 or endpoint:
            raise AssertionError("Unexpected draw interface")
        if high == 2:
            value = 0 if self.coin_calls == 0 else 1
            self.coin_calls += 1
            kind = "hidden_coin"
        elif high == 6:
            value, kind = 0, "hidden_permutation"
        elif high == self.stop:
            value = 1 if self.coefficient_calls == 0 else 0
            self.coefficient_calls += 1
            kind = "coefficient"
        else:
            raise AssertionError((low, high))
        self.events.append({"kind": kind, "value": value, "count": int(np.prod(size)) if size is not None else 1})
        return np.full(size if size is not None else (), value, dtype=dtype)


def main() -> None:
    rows = []
    for stop in (256, 257):
        for secret in (0, 255):
            rng = ScriptedDraws(stop)
            result = generator.generate(np.array([secret], dtype=np.uint8), 0, stop, rng, max_rounds=2)
            assert rng.coin_calls == 1 and rng.coefficient_calls == 2
            assert int(result["attempts"][0]) == 2
            assert int(result["shadow1"][0]) == secret and int(result["shadow2"][0]) == secret
            wrong_rng = ScriptedDraws(stop)
            try:
                generator.generate(np.array([secret], dtype=np.uint8), 0, stop, wrong_rng,
                                   max_rounds=2, negative_control_resample_pattern=True)
            except generator.GenerationExhaustedError:
                detected = True
            else:
                detected = False
            assert detected and wrong_rng.coin_calls == 2
            rows.append({"coefficient_stop": stop, "secret": secret,
                         "first_rejection_reason": "hidden_parity" if secret == 0 else "field_residue_256",
                         "fixed_state_events": rng.events, "wrong_state_events": wrong_rng.events,
                         "negative_control_detected_by_trace": detected})
    differences = []
    for stop in (256, 257):
        reference = channel_pair(stop)
        coefficients = np.arange(stop)
        for bit in (0, 1):
            for secret in range(256):
                outputs = generator.source_shares(secret, coefficients)
                wrong = {1: np.zeros(256), 2: np.zeros(256)}
                normalizer = 0.0
                for pattern, weight in trace_reference.PATTERN_LAWS[bit]:
                    accepted = generator.source_accepts(secret, coefficients, *pattern)
                    normalizer += float(weight) * int(accepted.sum())
                    for participant in (1, 2):
                        wrong[participant] += float(weight) * np.bincount(
                            outputs[participant - 1][accepted], minlength=256)
                for participant in (1, 2):
                    distribution = wrong[participant] / normalizer
                    correct = reference[participant][bit][secret]
                    differences.append({"coefficient_stop": stop, "public_bit": bit,
                                        "secret": secret, "participant": participant,
                                        "tv_wrong_order_vs_fixed_state": float(0.5 * np.abs(distribution - correct).sum()),
                                        "max_abs_cell_difference": float(np.max(np.abs(distribution - correct)))})
    strongest = max(differences, key=lambda row: row["tv_wrong_order_vs_fixed_state"])
    result = {"status": "completed", "timing": "post-hoc after frozen negative-control tests returned zero rejections",
              "not_a_replacement_for_preregistered_gof": True,
              "trace_checks": rows, "channel_rows_compared": len(differences),
              "maximum_wrong_order_tv_case": strongest,
              "rows_with_nonzero_wrong_order_tv_above_1e_14": sum(row["tv_wrong_order_vs_fixed_state"] > 1e-14 for row in differences),
              "interpretation": "Retry-order differences are measured deterministically; the original stochastic negative-control result remains unchanged"}
    write_json(Path(__file__).resolve().parents[2] / "outputs" / "tcsvt21_generator" / "posthoc_diagnostics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
