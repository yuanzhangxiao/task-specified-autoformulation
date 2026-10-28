"""Check fixed analytical controls for the Lyra/Vega task-selection discussion.

This is a short arithmetic proof check, not a fitting or acquisition campaign.
The identities and global arguments are in VEGA_PREMISE_REVISION_TASK_SELECTION.md.
An optional immutable receipt permits deterministic reuse after interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoformalism.research.premise_intervals import Interval, kernel  # noqa: E402

P = Interval.point
INTERVAL_SOURCE = ROOT / "src/autoformalism/research/premise_intervals.py"
INTERVAL_SHA256 = "b30340586bc890e6f8a3d0e1550bcf9a6e1432d7dc4caf161ba5a1c7a1bcd7e7"


def require(condition: bool, message: str) -> None:
    """Keep proof checks active even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def contains(value: Interval, point: str) -> bool:
    """Check numerical consistency with an analytically known exact value."""
    return value.lo <= Decimal(point) <= value.hi


def controls() -> dict:
    """Evaluate outward bounds; containment alone never proves an identity."""
    require(
        hashlib.sha256(INTERVAL_SOURCE.read_bytes()).hexdigest() == INTERVAL_SHA256,
        "The reviewed interval implementation changed; requalify this check.",
    )
    rate = P("0.2")
    eta = P("0.6").exp()
    require(eta.lo > Decimal("0.5") and eta.hi < 2, "Invalid pulse-scale witness")
    same_time = eta * P("-0.6").exp()
    require(contains(same_time, "1"), "Same-time identity is inconsistent")
    later = P("-0.4").exp()
    separation = (P(1) - later) * kernel(rate, Decimal(2))
    half_separation = separation / P(2)
    require(half_separation.lo > Decimal("0.1"), "The stated lower bound failed")

    # Allowing a separate scale at each run preserves the two-world ambiguity.
    unshared_scales = {}
    for time in (3, 5):
        scale = P("0.5") * (rate * P(time)).exp()
        require(scale.lo >= Decimal("0.5") and scale.hi <= 2, "Scale outside prior")
        normalized = scale * (-(rate * P(time))).exp()
        require(contains(normalized, "0.5"), "Unshared-scale identity failed")
        unshared_scales[str(time)] = scale.to_json()

    # A separate two-query control has predictions (x, x^2), x in [0.1, 1].
    # Its global minimax proof is analytical, not inferred from these evaluations.
    a = P("0.1")
    x = (P("5.4").sqrt() - P(1)) / P(2)
    free_risk = (P(1) - a.square()) / P(2)
    model_risk = x - a
    require(x.lo > Decimal("0.1") and x.hi < Decimal("0.9"), "Wrong root branch")
    balance = x.square() + x - P("1.1")
    require(contains(balance, "0"), "Minimax balance equation failed")
    require(model_risk.lo > free_risk.hi, "No strict two-query gap")

    # Approximate-premise thresholds are declared synthetic rates, not biology.
    c_l = c_r = P("0.02")
    width = rate - c_l - c_r
    require(contains(width, "0.16"), "Sharp width threshold changed")
    # [c_R, s-c_L] preserves each premise in a different world at equality.
    # Strictly smaller intervals cannot reach both threshold faces.
    return {
        "protocol": "premise-revision-selection-controls-v1",
        "source_sha256": {
            "premise_intervals.py": INTERVAL_SHA256,
            Path(__file__).name: hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "one_time_product": {
            "A": "1",
            "s": "0.2",
            "first_time": "3",
            "second_time": "5",
            "post_pulse_delay": "2",
            "second_world_eta": eta.to_json(),
            "same_time_factor_consistency": same_time.to_json(),
            "second_time_factor": later.to_json(),
            "future_mean_separation": separation.to_json(),
            "unrestricted_prediction_risk_lower_bound": half_separation.to_json(),
        },
        "unshared_scale_counterexample": unshared_scales,
        "two_query_model_restriction": {
            "a": "0.1",
            "x_star": x.to_json(),
            "free_vector_risk": free_risk.to_json(),
            "point_model_risk": model_risk.to_json(),
            "risk_gap": (model_risk - free_risk).to_json(),
            "units": "pulse increment divided by its common known positive kernel",
            "scope": "separate analytical family, not the four M2 cases",
        },
        "named_revision_width": {
            "s": "0.2",
            "c_L": "0.02",
            "c_R": "0.02",
            "strict_sufficient_width": width.to_json(),
            "sharp_boundary_interval": ["0.02", "0.18"],
        },
        "scope": (
            "deterministic analytical controls; "
            "no sampled data or biological validation"
        ),
    }


def save_receipt(path: Path, report: dict) -> str:
    """Publish atomically without overwriting; reuse only identical completed work."""
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == data, "Existing receipt differs; use a fresh path."
        )
        return "reused"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == data, "Concurrent receipt differs.")
            return "reused"
        return "written"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    """Check fixed controls and optionally preserve an immutable result receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = controls()
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = save_receipt(args.output, report)
        print(f"Verified analytical controls; receipt {status}.")


if __name__ == "__main__":
    main()
