"""Exact control for recovering latent functions from calibrated output balances."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.construction_functions import file_hash

DEFAULT = Path("output/anchored-latent-functions-20261126")
BASIS = ["1", "a", "b", "a^2", "a*b", "b^2"]


@dataclass(frozen=True)
class Observation:
    """Measured values for one unit-time interval at constant x and u."""

    x: Q
    u: Q
    r0: Q
    r1: Q
    s: Q
    y0: Q
    y1: Q


def simulate(x: Q, u: Q, initial: Q, *, latent_scale: Q = Q(1)) -> Observation:
    """Generate exact samples; expose only the two inputs and three readouts."""
    start = initial * latent_scale
    speed = (1 + x * u) * latent_scale
    end = start + speed
    latent_value = 2 + x * x - u
    # y' = 1 + z² + 2zx. Integrate with z(t) = initial + speed*t.
    area = (
        1
        + (start**2 + start * speed + speed**2 / 3) / latent_scale**2
        + x * (start + end) / latent_scale
    )
    y0 = 5 - x + u
    return Observation(
        x,
        u,
        x + start / latent_scale,
        x + end / latent_scale,
        u + latent_value,
        y0,
        y0 + area,
    )


def moments(a0: Q, a1: Q, b: Q) -> list[Q]:
    """Average a quadratic basis for linearly varying a and constant b."""
    a = (a0 + a1) / 2
    square = (a0 * a0 + a0 * a1 + a1 * a1) / 3
    return [Q(1), a, b, square, a * b, b * b]


def identify(design: list[list[Q]], values: list[Q]) -> dict:
    """Solve an overdetermined noiseless design exactly in the declared basis."""
    if not design:
        raise ValueError("At least one measured interval is required")
    width = len(BASIS)
    aug = [[*d, v] for d, v in zip(design, values, strict=True)]
    rank = 0
    for j in range(width):
        pivot = next((i for i in range(rank, len(aug)) if aug[i][j]), None)
        if pivot is None:
            continue
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        aug[rank] = [v / aug[rank][j] for v in aug[rank]]
        for i, row in enumerate(aug):
            if i != rank and row[j]:
                aug[i] = [v - row[j] * a for v, a in zip(row, aug[rank], strict=True)]
        rank += 1
    if any(all(v == 0 for v in row[:-1]) and row[-1] for row in aug):
        return {"status": "family_excluded", "rank": rank}
    if rank < width:
        return {"status": "insufficient_variation", "rank": rank}
    return {
        "status": "unique_in_declared_family",
        "rank": rank,
        "coefficients": [str(row[-1]) for row in aug[:width]],
    }


def recover(rows: list[Observation], *, calibrated: bool) -> dict:
    """Recover three maps using the known readouts and the total-degree-two class."""
    inputs = [moments(r.x, r.x, r.u) for r in rows]
    result = {"w=h(x,u)": identify(inputs, [r.s - r.u for r in rows])}
    if calibrated:
        z = [(r.r0 - r.x, r.r1 - r.x) for r in rows]
        result["z'=f(x,u)"] = identify(inputs, [b - a for a, b in z])
        design = [moments(a, b, r.x) for (a, b), r in zip(z, rows, strict=True)]
        result["y'-1=H(z,x)"] = identify(design, [r.y1 - r.y0 - 1 for r in rows])
    else:
        for name in ("z'=f(x,u)", "y'-1=H(z,x)"):
            result[name] = {"status": "unknown_latent_scale"}
    return result


def run(root: Path) -> dict:
    """Run three resumable exact controls on the same declared functional families."""
    plan = sealed_write(
        root / "plan.json",
        {
            "basis": BASIS,
            "cases": ["crossed", "same_input", "unknown_gain"],
            "source_sha256": file_hash(Path(__file__)),
        },
    )
    summary = {}
    for name in plan["cases"]:
        path = root / f"{name}.json"
        if path.exists():
            record = sealed_read(path)
            if (record["plan"], record["case"]) != (plan["artifact_sha256"], name):
                raise ValueError("Control checkpoint identity differs")
        else:
            events = [
                (Q(x), Q(u), Q(z0))
                for x, u, z0 in product(range(3), range(3), (1, 3))
                if name != "same_input" or x == u
            ]
            rows = [simulate(*args) for args in events]
            extra = {}
            if name == "unknown_gain":
                scaled = [simulate(*args, latent_scale=Q(2)) for args in events]
                extra["identical_events_after_scaling_z_by_2"] = sum(
                    a == b for a, b in zip(rows, scaled, strict=True)
                )
            record = sealed_write(
                path,
                {
                    "plan": plan["artifact_sha256"],
                    "case": name,
                    "events": len(rows),
                    "functions": recover(rows, calibrated=name != "unknown_gain"),
                    **extra,
                },
            )
        summary[name] = record
    return sealed_write(
        root / "summary.json", {"plan": plan["artifact_sha256"], "cases": summary}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with _lock(args.root):
        print(run(args.root)["artifact_sha256"])
