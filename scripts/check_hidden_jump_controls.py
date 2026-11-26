"""Check an additive hidden-nuisance hypothesis on sampled, public control events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import hidden_jump_contrasts as jump
from autoformalism.staged_topology import content_hash
from scripts.construction_functions import file_hash

DEFAULT = Path(__file__).resolve().parents[1] / "output/partial-shape-20261126/jumps"


def event(
    index: int,
    levels: list[float],
    *,
    interacting: bool = False,
    drift: float = 0.0,
    noise: float = 0.0,
) -> dict:
    """Integrate z'=drift and y'=z+u² or y'=z u; export only sampled y and u."""
    hidden_initial = 2.0**index
    dt = 0.25
    t = [i * dt for i in range(len(levels))]
    y = [float(index)]
    for t0, t1, u in zip(t[:-1], t[1:], levels[:-1], strict=True):
        mean_z = hidden_initial + drift * (t0 + t1) / 2
        mean_rate = mean_z * u if interacting else mean_z + u**2
        y.append(y[-1] + dt * mean_rate)
    return {
        "trajectory_id": f"train_{index:03}",
        "time": t,
        "inputs": {"u": levels},
        "targets": {"y": [v + (-1) ** i * noise for i, v in enumerate(y)]},
        "covariates": {},
        "auxiliaries": {},
    }


def controls() -> dict:
    """Predefine identifiable, inconsistent, bounded and unavailable contrasts."""
    transitions = [(0, 1), (1, 2), (2, 3), (3, 0)]
    default = {
        "target": "y",
        "source": "u",
        "interpolation": {"u": jump.STEP},
        "observation_error": 0.0,
        "curvature_bound": 0.0,
        "continuous_nuisance": True,
    }
    exact = [event(i, [a, b, b]) for i, (a, b) in enumerate(transitions)]
    mixed = [
        event(i, [a, b, b], interacting=True) for i, (a, b) in enumerate(transitions)
    ]
    budget = {"curvature_bound": 0.0625, "observation_error": 2**-10}
    noisy = [
        event(
            i,
            [a, b, b],
            drift=(-1) ** i * budget["curvature_bound"],
            noise=budget["observation_error"],
        )
        for i, (a, b) in enumerate(transitions)
    ]
    simultaneous = [
        {**r, "inputs": {**r["inputs"], "u2": r["inputs"]["u"]}} for r in exact
    ]
    return {
        "exact": (exact, default),
        "bounded": (noisy, default | budget),
        "interaction": (mixed, default),
        "disconnected": ([exact[0], exact[2]], default),
        "continuous_input": (exact, default | {"interpolation": {"u": jump.LINEAR}}),
        "simultaneous": (
            simultaneous,
            default | {"interpolation": {"u": jump.STEP, "u2": jump.STEP}},
        ),
        "uncertified_rates": (exact, default | {"curvature_bound": None}),
        "marked_resets": (
            exact,
            default | {"reset_knots": {r["trajectory_id"]: [1] for r in exact}},
        ),
    }


def run(root: Path) -> dict:
    """Seal each control independently and reuse matching results on resume."""
    cases = controls()
    with _lock(root / "write-lock"):
        plan = sealed_write(
            root / "plan.json",
            {
                "protocol": "hidden-additive-input-jump-controls-1",
                "inputs": {k: content_hash([r, s]) for k, (r, s) in cases.items()},
                "code_sha256": {
                    p.name: file_hash(p) for p in (Path(__file__), Path(jump.__file__))
                },
            },
        )
        summary = {}
        for name, (rows, kwargs) in cases.items():
            path = root / "cases" / f"{name}.json"
            if path.exists():
                item = sealed_read(path)
            else:
                item = sealed_write(
                    path,
                    {
                        "plan": plan["artifact_sha256"],
                        "name": name,
                        "result": jump.screen_input_jumps(rows, **kwargs),
                    },
                )
            if (item["plan"], item["name"]) != (plan["artifact_sha256"], name):
                raise ValueError("Saved jump control differs from frozen design")
            summary[name] = {
                "status": item["result"]["status"],
                "receipt": item["artifact_sha256"],
                "witnesses": len(item["result"]["witnesses"]),
            }
        return sealed_write(
            root / "summary.json", {"plan": plan["artifact_sha256"], "cases": summary}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    print(json.dumps(run(parser.parse_args().root), indent=2))
