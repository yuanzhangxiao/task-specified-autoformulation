"""Exact rational checks for the premise exploration applicability audit.

These checks support analytical arguments; they are not acquisition experiments
or a simulation of Track-and-Stop. Existing receipts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_premise_revision_controls import save_receipt  # noqa: E402

Q = Fraction


def retained_radius(values: tuple[Fraction, ...]) -> Fraction:
    """Compute the exact scalar minimax radius with centers among the values."""
    if not values or any(type(value) is not Fraction for value in values):
        raise ValueError("provide a nonempty tuple of exact Fractions")
    return min(max(abs(center - value) for value in values) for center in values)


def regularity_boundary(radius: Fraction) -> dict[str, str]:
    """Check the small-radius branch of the analytical uniformity counterexample."""
    if type(radius) is not Fraction or not 0 < radius <= Q(1, 100):
        raise ValueError("radius must be an exact Fraction in (0, 1/100]")
    tolerance, mean = Q(1, 10), Q(1, 20)
    answer = tolerance + radius / 2
    left_threshold = answer - tolerance
    right_threshold = answer + tolerance
    common_threshold = answer + radius + tolerance
    singleton_divergence = (mean - left_threshold) ** 2 / 2
    common_divergence = (common_threshold - mean) ** 2 / 2
    gap = common_divergence - singleton_divergence
    conditions = (
        abs(answer - mean) < tolerance,
        Q(-1, 10) < answer - radius < answer + radius < Q(11, 10),
        answer - radius - tolerance < 0,
        0 < left_threshold < mean < right_threshold < common_threshold < 1,
        mean - left_threshold < right_threshold - mean,
        gap == Q(1, 100) + radius / 4 + radius**2,
        gap > Q(1, 100),
    )
    if not all(conditions):
        raise ValueError("analytical boundary conditions failed")
    return {
        "radius": str(radius),
        "strictly_correct_answer": str(answer),
        "true_mean": str(mean),
        "singleton_left_threshold": str(left_threshold),
        "singleton_right_threshold": str(right_threshold),
        "neighborhood_common_threshold": str(common_threshold),
        "singleton_divergence": str(singleton_divergence),
        "neighborhood_divergence": str(common_divergence),
        "gap": str(gap),
    }


def report() -> dict:
    """Evaluate fixed exact examples and bind the checker and receipt helper."""
    old_values, new_values = (Q(0), Q(1), Q(2)), (Q(0), Q(2))
    old_radius, new_radius = retained_radius(old_values), retained_radius(new_values)
    if (old_radius, new_radius) != (Q(1), Q(2)):
        raise ValueError("retained-center counterexample failed")
    source_paths = (
        Path(__file__),
        ROOT / "scripts/check_premise_revision_controls.py",
        ROOT / "src/autoformalism/research/premise_intervals.py",
    )
    return {
        "protocol": "premise-exploration-contract-check-v1",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "retained_center": {
            "old_values": [str(value) for value in old_values],
            "new_values": [str(value) for value in new_values],
            "old_radius": str(old_radius),
            "new_radius": str(new_radius),
            "fixed_center_one_radius_before_and_after": "1",
        },
        "regularity_boundary": {
            "mean_domain": ["0", "1"],
            "answer_domain": ["-1/10", "11/10"],
            "gaussian_variance": "1",
            "prediction_tolerance": "1/10",
            "failed_uniformity_tolerance": "1/200",
            "examples": [
                regularity_boundary(Q(1, denominator))
                for denominator in (100, 10000, 100000000)
            ],
        },
        "scope": (
            "Exact rational arithmetic checks. The all-radius proof is in the "
            "audit document; finite examples alone do not prove it. "
            "No biological or acquisition-performance claim."
        ),
    }


def main() -> None:
    """Print the exact checks or atomically publish/reuse their immutable receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report()
    if args.output is None:
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        status = save_receipt(args.output, result)
        print(f"Exact contract checks passed; receipt {status}.")


if __name__ == "__main__":
    main()
