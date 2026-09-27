"""Independent CPU implementation of the TCSVT21 (2, 2) source rules.

Source: X. Yan et al., "A Common Method of Share Authentication in Image
Secret Sharing," IEEE TCSVT 31(7), 2896--2908, DOI 10.1109/TCSVT.2020.3025527.
Primary text: https://www.researchgate.net/publication/345333246_A_Common_Method_of_Share_Authentication_in_Image_Secret_Sharing

Equation (1), p. 2899, supplies the RG-VSS primitive. Algorithm 1, p. 2900,
supplies the per-pixel duplicate-and-permute construction, polynomial, and
rejection loop. These rules are implemented directly, independently of the
enumerated observation channel.

The source calls polynomial coefficients random, later describes them as
grayscale, and counts 256**(k-1) coefficient choices (p. 2901). Consequently
``coefficient_stop=256`` represents the byte-domain reading; ``257`` is an
explicit full-field sensitivity case. Uniform integer draws and uniform
permutations implement the source's random choices. The source does not
specify a PRNG or explicitly state a uniform coefficient distribution.
"""

from __future__ import annotations

from itertools import permutations
from numbers import Integral

import numpy as np


__all__ = [
    "GenerationExhaustedError",
    "generate",
    "hidden_from_primitives",
    "source_shares",
    "source_parity",
    "source_accepts",
]


class GenerationExhaustedError(RuntimeError):
    """Generation stopped with unfinished pixels; no share result is returned.

    ``attempts`` and ``pending_mask`` preserve per-pixel diagnostics in the
    original shape. They are copies, so the exception retains the failure
    state without exposing partially generated shadow arrays as valid output.
    """

    def __init__(
        self,
        attempts: np.ndarray,
        pending_mask: np.ndarray,
        coefficient_stop: int,
        max_rounds: int,
        negative_control_resample_pattern: bool,
    ) -> None:
        self.attempts = attempts.copy()
        self.pending_mask = pending_mask.copy()
        self.coefficient_stop = coefficient_stop
        self.max_rounds = max_rounds
        self.negative_control_resample_pattern = negative_control_resample_pattern
        count = int(np.count_nonzero(pending_mask))
        super().__init__(
            f"TCSVT21 generation incomplete: {count} pixel(s) still pending "
            f"after {max_rounds} coefficient attempts per pending pixel "
            f"with coefficient_stop={coefficient_stop}; no share result returned."
        )


def _integer_values(values: np.ndarray, name: str, upper: int) -> np.ndarray:
    """Validate integer representatives before converting to safe arithmetic."""
    array = np.asarray(values)
    if array.dtype.kind not in "bui":
        raise TypeError(f"{name} must contain integer values")
    if np.any(array < 0) or np.any(array > upper):
        raise ValueError(f"{name} must lie in [0, {upper}]")
    return array.astype(np.int64, copy=False)


def _binary_values(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "buif":
        raise TypeError(f"{name} must contain numeric binary values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.uint8, copy=False)


def _low_four_parity(values: np.ndarray) -> np.ndarray:
    """Extract the four bits arithmetically, without a lookup or bit shifts."""
    bit0 = values % 2
    bit1 = (values // 2) % 2
    bit2 = (values // 4) % 2
    bit3 = (values // 8) % 2
    return np.asarray((bit0 + bit1 + bit2 + bit3) % 2, dtype=np.uint8)


def source_shares(
    secret: np.ndarray, coefficient: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return f(1), f(2) for f(x) = (secret + coefficient*x) mod 257.

    Inputs broadcast. ``secret`` contains integers 0..255 and ``coefficient``
    integers 0..256. Outputs are uint16 so that the forbidden value 256 is
    retained for the caller's acceptance check, never wrapped to a byte.
    """
    secret_values = _integer_values(secret, "secret", 255)
    coefficient_values = _integer_values(coefficient, "coefficient", 256)
    first = (secret_values + coefficient_values) % 257
    second = (secret_values + 2 * coefficient_values) % 257
    return np.asarray(first, dtype=np.uint16), np.asarray(second, dtype=np.uint16)


def source_parity(values: np.ndarray) -> np.ndarray:
    """Return XOR of bits 0, 1, 2, 3 for integer representatives 0..256.

    The parity of 256 is zero, but 256 is nevertheless rejected separately by
    Algorithm 1's byte-range condition. Output dtype is uint8.
    """
    return _low_four_parity(_integer_values(values, "values", 256))


def source_accepts(
    secret: np.ndarray,
    coefficient: np.ndarray,
    pattern1: np.ndarray,
    pattern2: np.ndarray,
) -> np.ndarray:
    """Pure, broadcasting predicate for Algorithm 1 Step 4, with boolean output."""
    first, second = source_shares(secret, coefficient)
    first_pattern = _binary_values(pattern1, "pattern1")
    second_pattern = _binary_values(pattern2, "pattern2")
    return np.asarray(
        (first < 256)
        & (second < 256)
        & (_low_four_parity(first) == first_pattern)
        & (_low_four_parity(second) == second_pattern),
        dtype=np.bool_,
    )


def hidden_from_primitives(
    auth: np.ndarray,
    fair_bit: np.ndarray,
    permutation_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct Step 2's pattern1, pattern2, auxiliary from explicit draws.

    Inputs broadcast. Auth and fair_bit are binary. Permutation indices 0..5
    mean (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0), respectively.
    These permutations reorder the labeled triple (b, b XOR auth, b). The
    index table contains only permutations, not hidden-pattern probabilities.
    This pure helper allows exhaustive checking of all primitive draw states.
    """
    auth_values = _binary_values(auth, "auth")
    coin_values = _binary_values(fair_bit, "fair_bit")
    index_values = _integer_values(permutation_index, "permutation_index", 5)
    auth_values, coin_values, index_values = np.broadcast_arrays(
        auth_values, coin_values, index_values
    )
    second_bit = np.bitwise_xor(coin_values, auth_values)
    triples = np.stack((coin_values, second_bit, coin_values), axis=-1)
    position_orders = np.asarray(tuple(permutations((0, 1, 2))), dtype=np.int64)
    shuffled = np.take_along_axis(triples, position_orders[index_values], axis=-1)
    return shuffled[..., 0], shuffled[..., 1], shuffled[..., 2]


def _draw_hidden_triples(
    auth_flat: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Perform Step 2 separately at each pixel using primitive random draws."""
    count = auth_flat.size
    first_bit = rng.integers(0, 2, size=count, dtype=np.uint8)
    # Every pixel gets a separate uniform draw from all six labeled
    # permutations, even when repeated bit values yield equal output triples.
    permutation_index = rng.integers(0, 6, size=count, dtype=np.uint8)
    hidden_parts = hidden_from_primitives(auth_flat, first_bit, permutation_index)
    return np.stack(hidden_parts, axis=1)


def generate(
    secret: np.ndarray,
    auth: np.ndarray,
    coefficient_stop: int,
    rng: np.random.Generator,
    max_rounds: int = 10000,
    *,
    negative_control_resample_pattern: bool = False,
) -> dict[str, np.ndarray]:
    """Generate two byte shadows and one binary auxiliary share on the CPU.

    ``secret`` must be a uint8 ndarray of any shape, including scalar or empty
    arrays. ``auth`` contains exact 0/1 values and must broadcast to that shape.
    Uniform coefficients are drawn from ``range(coefficient_stop)`` where the
    exclusive stop is 256 (byte reading) or 257 (full-field sensitivity).

    Hidden triples are drawn once per pixel, before any coefficient attempts.
    Every round draws a fresh coefficient for each pending pixel. Rejection
    returns to Step 3 and preserves Step 2's entire hidden triple, including
    its auxiliary bit. Pixels already accepted receive no additional draws.

    Returned keys are ``shadow1``, ``shadow2``, ``auxiliary``, ``pattern1``, and
    ``pattern2`` (uint8), ``coefficient`` (accepted coefficient, uint16), and
    ``attempts`` (int64, including the accepted attempt). Every value has the
    original secret shape. The input arrays are not modified.

    ``max_rounds`` is a positive per-pixel attempt limit, not a fallback rule.
    Any unfinished pixels cause GenerationExhaustedError with diagnostics.

    Setting ``negative_control_resample_pattern=True`` redraws the hidden
    triple after rejection as a negative control. Algorithm 1 keeps it fixed.
    """
    if not isinstance(secret, np.ndarray) or secret.dtype != np.dtype(np.uint8):
        raise TypeError("secret must be a numpy ndarray with dtype uint8")
    if (
        isinstance(coefficient_stop, (bool, np.bool_))
        or not isinstance(coefficient_stop, Integral)
        or coefficient_stop not in (256, 257)
    ):
        raise ValueError("coefficient_stop must be the integer 256 or 257")
    if (
        isinstance(max_rounds, (bool, np.bool_))
        or not isinstance(max_rounds, Integral)
        or max_rounds <= 0
    ):
        raise ValueError("max_rounds must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(negative_control_resample_pattern, (bool, np.bool_)):
        raise TypeError("negative_control_resample_pattern must be boolean")

    shape = secret.shape
    secret_flat = secret.reshape(-1)
    auth_values = _binary_values(auth, "auth")
    try:
        auth_flat = np.broadcast_to(auth_values, shape).reshape(-1)
    except ValueError as error:
        raise ValueError("auth must broadcast to the secret array shape") from error

    count = secret_flat.size
    hidden = _draw_hidden_triples(auth_flat, rng)
    shadow1 = np.empty(count, dtype=np.uint8)
    shadow2 = np.empty(count, dtype=np.uint8)
    coefficient = np.empty(count, dtype=np.uint16)
    attempts = np.zeros(count, dtype=np.int64)
    pending = np.arange(count)

    for round_index in range(int(max_rounds)):
        if pending.size == 0:
            break
        if negative_control_resample_pattern and round_index > 0:
            hidden[pending] = _draw_hidden_triples(auth_flat[pending], rng)

        proposed = rng.integers(
            0, int(coefficient_stop), size=pending.size, dtype=np.uint16
        )
        first, second = source_shares(secret_flat[pending], proposed)
        attempts[pending] += 1
        accepted = (
            (first < 256)
            & (second < 256)
            & (_low_four_parity(first) == hidden[pending, 0])
            & (_low_four_parity(second) == hidden[pending, 1])
        )
        accepted_pixels = pending[accepted]
        shadow1[accepted_pixels] = first[accepted].astype(np.uint8)
        shadow2[accepted_pixels] = second[accepted].astype(np.uint8)
        coefficient[accepted_pixels] = proposed[accepted]
        pending = pending[~accepted]

    if pending.size:
        pending_mask = np.zeros(count, dtype=np.bool_)
        pending_mask[pending] = True
        raise GenerationExhaustedError(
            attempts.reshape(shape),
            pending_mask.reshape(shape),
            int(coefficient_stop),
            int(max_rounds),
            bool(negative_control_resample_pattern),
        )

    return {
        "shadow1": shadow1.reshape(shape),
        "shadow2": shadow2.reshape(shape),
        "auxiliary": hidden[:, 2].copy().reshape(shape),
        "attempts": attempts.reshape(shape),
        "coefficient": coefficient.reshape(shape),
        "pattern1": hidden[:, 0].copy().reshape(shape),
        "pattern2": hidden[:, 1].copy().reshape(shape),
    }
