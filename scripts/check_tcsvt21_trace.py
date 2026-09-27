from __future__ import annotations

from fractions import Fraction


FIELD = 257
WITNESS_SECRETS = (3, 252)
PATTERN_LAWS = {
    0: (((0, 0), Fraction(1, 2)), ((1, 1), Fraction(1, 2))),
    1: (
        ((0, 0), Fraction(1, 6)),
        ((1, 1), Fraction(1, 6)),
        ((0, 1), Fraction(1, 3)),
        ((1, 0), Fraction(1, 3)),
    ),
}


def xor4(value: int) -> int:
    return (value ^ (value >> 1) ^ (value >> 2) ^ (value >> 3)) & 1


def shares(secret: int, coefficient: int) -> tuple[int, int]:
    return (
        (secret + coefficient) % FIELD,
        (secret + 2 * coefficient) % FIELD,
    )


def accepted_coefficients(
    secret: int, pattern: tuple[int, int], candidates: range
) -> tuple[int, ...]:
    accepted: list[int] = []
    for coefficient in candidates:
        x1, x2 = shares(secret, coefficient)
        if x1 >= 256 or x2 >= 256:
            continue
        if (xor4(x1), xor4(x2)) == pattern:
            accepted.append(coefficient)
    return tuple(accepted)


def participant_row(
    secret: int, bit: int, candidates: range, participant: int
) -> dict[int, Fraction]:
    if participant not in (1, 2):
        raise ValueError("participant must be 1 or 2")
    row: dict[int, Fraction] = {}
    for pattern, pattern_mass in PATTERN_LAWS[bit]:
        accepted = accepted_coefficients(secret, pattern, candidates)
        if not accepted:
            raise AssertionError(
                f"empty accepted set for bit={bit}, secret={secret}, "
                f"pattern={pattern}"
            )
        mass = pattern_mass / len(accepted)
        for coefficient in accepted:
            observed = shares(secret, coefficient)[participant - 1]
            row[observed] = row.get(observed, Fraction(0)) + mass
    if sum(row.values(), Fraction(0)) != 1:
        raise AssertionError(f"channel row for secret={secret} is not normalized")
    return row


def check_all_accepted_sets(candidates: range) -> tuple[int, int, int]:
    sizes: list[int] = []
    for bit, pattern_law in PATTERN_LAWS.items():
        for secret in range(256):
            for pattern, _ in pattern_law:
                accepted = accepted_coefficients(secret, pattern, candidates)
                if not accepted:
                    raise AssertionError(
                        f"empty accepted set for bit={bit}, secret={secret}, "
                        f"pattern={pattern}"
                    )
                sizes.append(len(accepted))
                for coefficient in accepted:
                    x1, x2 = shares(secret, coefficient)
                    if (2 * x1 - x2) % FIELD != secret:
                        raise AssertionError(
                            f"reconstruction failed for secret={secret}, "
                            f"coefficient={coefficient}"
                        )
    return len(sizes), min(sizes), max(sizes)


def find_disjoint_pair(
    bit: int, candidates: range, participant: int
) -> tuple[int, int, dict[int, Fraction], dict[int, Fraction]]:
    rows = [participant_row(secret, bit, candidates, participant) for secret in range(256)]
    supports = [set(row) for row in rows]
    for left in range(256):
        for right in range(left + 1, 256):
            if supports[left].isdisjoint(supports[right]):
                return left, right, rows[left], rows[right]
    raise AssertionError(f"no disjoint support pair for participant {participant}")


def check_reading(name: str, candidates: range) -> None:
    print(f"[{name}]")
    groups, smallest, largest = check_all_accepted_sets(candidates)
    print(
        f"all_positive_groups={groups}, accepted_size_range=[{smallest}, {largest}]"
    )
    participant_one_rows = {}
    for secret in WITNESS_SECRETS:
        accepted_sizes = {}
        for pattern, _ in PATTERN_LAWS[0]:
            accepted = accepted_coefficients(secret, pattern, candidates)
            accepted_sizes[str(pattern)] = len(accepted)
            for coefficient in accepted:
                x1, x2 = shares(secret, coefficient)
                recovered = (2 * x1 - x2) % FIELD
                if recovered != secret:
                    raise AssertionError(
                        f"reconstruction failed for secret={secret}, coefficient={coefficient}"
                    )
        row = participant_row(secret, 0, candidates, 1)
        participant_one_rows[secret] = row
        print(
            f"secret={secret}: accepted_sizes={accepted_sizes}, "
            f"participant1_support={sorted(row)}"
        )

    left = set(participant_one_rows[WITNESS_SECRETS[0]])
    right = set(participant_one_rows[WITNESS_SECRETS[1]])
    if left & right:
        raise AssertionError("witness supports are not disjoint")
    print("result: disjoint participant-1 supports, exact TV = 1, reconstruction = pass")
    p2_left, p2_right, p2_left_row, p2_right_row = find_disjoint_pair(
        0, candidates, 2
    )
    print(
        f"participant2_witness=({p2_left}, {p2_right}), "
        f"supports=({sorted(p2_left_row)}, {sorted(p2_right_row)})"
    )
    print("result: disjoint participant-2 supports, exact TV = 1, reconstruction = pass")


def main() -> None:
    check_reading("full-field", range(257))
    check_reading("byte-domain", range(256))


if __name__ == "__main__":
    main()
