"""Frozen synthetic premise-revision control; no fitting or benchmark access.

Inference sees observed means and a public nine-world catalog. Only the evaluator
sees generating-world labels. Checkpoints preserve raw replicates and the original
sequential PCG64 schedule at world boundaries. Hashes detect accidental changes;
they are not signatures against a party able to rewrite the whole output root.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta, norm

PROTOCOL = "vega-premise-finite-reference-v1"
REPLICATES = 8
ALPHA = 0.05
SIGMA_X = 0.05
SIGMA_LIGAND = 0.1
TIME = 5.0
Z = float(norm.ppf(1 - ALPHA / 4))
RADIUS_X = Z * SIGMA_X / math.sqrt(REPLICATES)
RADIUS_LIGAND = Z * SIGMA_LIGAND / math.sqrt(REPLICATES)
PACKET_SHA256 = "8da2b8d5fcccebe72fed6c01272762e1f17e884e124d9ce0b01384ddd64c13ee"


@dataclass(frozen=True)
class ReferenceConfig:
    """Only simulation count and seed vary; scientific settings remain frozen."""

    trials_per_world: int = 5000
    seed: int = 20261028

    def __post_init__(self) -> None:
        if type(self.trials_per_world) is not int or self.trials_per_world <= 0:
            raise ValueError("trials_per_world must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True)
class World:
    """A public candidate, not a generating-world label passed to inference."""

    world_id: int
    kappa: float
    gamma: float
    sum_index: int
    mu_x: float
    mu_ligand: float
    premise_bits: int


@dataclass(frozen=True)
class Claims:
    """Universal conclusions, with all compatible violation patterns retained."""

    support_bits: int
    reject_bits: int
    conflict_bits: int
    violation_patterns: tuple[int, ...]


def response(total_rate: float, time: float = TIME, b: float = 1.0) -> float:
    """Unit-scale response kernel, stable at equal rates and large rate gaps."""
    for name, value in (("total_rate", total_rate), ("time", time), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite real number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if b == 0:
        raise ValueError("b must be strictly positive")
    gap = abs(b - total_rate)
    scaled_gap = gap * time
    if scaled_gap < 1:
        # Divide before multiplying by time: subnormal gap*time must not erase
        # a finite response. This also handles exactly equal rates and t=0.
        factor = 1.0 if scaled_gap == 0 else -math.expm1(-scaled_gap) / scaled_gap
        return time * math.exp(-min(b, total_rate) * time) * factor
    return math.exp(-min(b, total_rate) * time) * (-math.expm1(-gap * time)) / gap


def catalog() -> tuple[World, ...]:
    """Canonical integer sum indices preserve exact equal-sum equivalence."""
    return tuple(
        World(
            3 * ki + gi,
            ki / 10,
            gi / 10,
            ki + gi,
            response((ki + gi) / 10),
            math.exp(-ki / 10 * TIME),
            int(ki == 0) + 2 * int(gi == 0),
        )
        for ki in range(3)
        for gi in range(3)
    )


def _mask(mask: int) -> None:
    if type(mask) is not int or not 0 <= mask < 512:
        raise ValueError("retained mask must be an integer in [0, 511]")


def project(mask: int) -> Claims:
    """Project predicates directly; an empty set supplies no scientific claim."""
    _mask(mask)
    worlds = [w for w in catalog() if mask & (1 << w.world_id)]
    if not worlds:
        return Claims(0, 0, 0, ())
    support = sum(p for p in (1, 2) if all(w.premise_bits & p for w in worlds))
    reject = sum(p for p in (1, 2) if all(not w.premise_bits & p for w in worlds))
    conflict = sum(
        1 << (subset - 1)
        for subset in (1, 2, 3)
        if not any(w.premise_bits & subset == subset for w in worlds)
    )
    return Claims(
        support,
        reject,
        conflict,
        tuple(sorted({3 ^ w.premise_bits for w in worlds})),
    )


def diagnosis(mask: int) -> Claims:
    """Matched baseline via complete diagnoses and hitting sets, no minimization."""
    _mask(mask)
    patterns = sorted({3 ^ w.premise_bits for w in catalog() if mask >> w.world_id & 1})
    if not patterns:
        return Claims(0, 0, 0, ())
    union, intersection = 0, 3
    for pattern in patterns:
        union |= pattern
        intersection &= pattern
    conflicts = sum(
        1 << (subset - 1)
        for subset in (1, 2, 3)
        if all(subset & pattern for pattern in patterns)
    )
    return Claims(3 ^ union, intersection, conflicts, tuple(patterns))


def classify(
    x_means: ArrayLike, ligand_means: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Apply the prespecified two-stage intervals without accessing ground truth."""
    x, ligand = np.asarray(x_means, dtype=float), np.asarray(ligand_means, dtype=float)
    if x.ndim != 1 or x.shape != ligand.shape:
        raise ValueError("sample means must be one-dimensional with equal shape")
    if not np.isfinite(x).all() or not np.isfinite(ligand).all():
        raise ValueError("sample means must be finite")
    stage1, stage2 = (
        np.zeros(x.shape, dtype=np.int64),
        np.zeros(x.shape, dtype=np.int64),
    )
    for w in catalog():
        keep_x = np.abs(x - w.mu_x) <= RADIUS_X
        stage1 |= keep_x.astype(np.int64) << w.world_id
        keep_both = keep_x & (np.abs(ligand - w.mu_ligand) <= RADIUS_LIGAND)
        stage2 |= keep_both.astype(np.int64) << w.world_id
    return stage1, stage2


def _false_claims(claim: Claims, truth_bits: int) -> tuple[bool, bool, bool]:
    """Evaluator-only truth comparison: support, rejection, joint conflict."""
    return (
        bool(claim.support_bits & (3 ^ truth_bits)),
        bool(claim.reject_bits & truth_bits),
        any(
            claim.conflict_bits & (1 << (subset - 1)) and truth_bits & subset == subset
            for subset in (1, 2, 3)
        ),
    )


def score_masks(stage1: ArrayLike, stage2: ArrayLike, truth_id: int) -> dict[str, int]:
    """Score datasets, retaining empty sets in every denominator."""
    if type(truth_id) is not int or not 0 <= truth_id < 9:
        raise ValueError("truth_id must be an integer in [0, 8]")
    first, second = np.asarray(stage1), np.asarray(stage2)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("mask arrays must be one-dimensional with equal shape")
    for values in (first, second):
        if values.dtype.kind not in "iu" or np.any((values < 0) | (values >= 512)):
            raise ValueError("invalid retained mask array")
    if np.any((second & first) != second):
        raise ValueError("stage 2 must be a subset of stage 1")
    truth = catalog()[truth_id].premise_bits
    errors = np.array([_false_claims(project(i), truth) for i in range(512)])
    both_errors = errors[first] | errors[second]
    true_bit = 1 << truth_id
    singleton = (second > 0) & ((second & (second - 1)) == 0)
    metrics = {
        "stage1_true_retained": first & true_bit != 0,
        "stage2_true_retained": second & true_bit != 0,
        "stage1_empty": first == 0,
        "stage2_empty": second == 0,
        "final_correct_singleton": second == true_bit,
        "final_wrong_singleton": singleton & (second != true_bit),
        "any_false_support": both_errors[:, 0],
        "any_false_rejection": both_errors[:, 1],
        "any_false_conflict": both_errors[:, 2],
        "any_false_claim": both_errors.any(axis=1),
    }
    return {name: int(np.count_nonzero(value)) for name, value in metrics.items()}


def _normal_interval(center: float, radius: float, mean: float, sd: float) -> float:
    """Compute Gaussian interval probability without upper-tail cancellation."""
    lo, hi = (center - radius - mean) / sd, (center + radius - mean) / sd
    return float(norm.sf(lo) - norm.sf(hi) if lo >= 0 else norm.cdf(hi) - norm.cdf(lo))


def analytic_reference() -> list[dict[str, Any]]:
    """Integrate disjoint acceptance rectangles, using floating-point Gaussian CDFs.

    These are exact probability formulas for the frozen Gaussian experiment, not
    validated interval-arithmetic certificates or a guarantee outside its envelope.
    """
    worlds = catalog()
    groups = [[w for w in worlds if w.sum_index == s] for s in range(5)]
    x_centers = sorted({w.mu_x for w in worlds})
    ligand_centers = sorted({w.mu_ligand for w in worlds})
    for centers, radius in ((x_centers, RADIUS_X), (ligand_centers, RADIUS_LIGAND)):
        if any(right - left <= 2 * radius for left, right in pairwise(centers)):
            raise ValueError("analytic reference requires disjoint channel intervals")
    rows = []
    for truth in worlds:
        p_x = [
            _normal_interval(g[0].mu_x, RADIUS_X, truth.mu_x, SIGMA_X / math.sqrt(8))
            for g in groups
        ]
        p_ligand = [
            _normal_interval(
                w.mu_ligand,
                RADIUS_LIGAND,
                truth.mu_ligand,
                SIGMA_LIGAND / math.sqrt(8),
            )
            for w in worlds
        ]
        false = 0.0
        for group, px in zip(groups, p_x, strict=True):
            mask = sum(1 << w.world_id for w in group)
            if any(_false_claims(project(mask), truth.premise_bits)):
                false += px
            else:
                false += px * sum(
                    p_ligand[w.world_id]
                    for w in group
                    if w.premise_bits != truth.premise_bits
                )
        singleton = [p_x[w.sum_index] * p_ligand[w.world_id] for w in worlds]
        rows.append(
            {
                "world_id": truth.world_id,
                "any_false_claim_probability": false,
                "stage1_empty_probability": 1 - sum(p_x),
                "stage2_empty_probability": 1 - sum(singleton),
                "final_correct_singleton_probability": singleton[truth.world_id],
                "final_wrong_singleton_probability": sum(
                    p for i, p in enumerate(singleton) if i != truth.world_id
                ),
            }
        )
    return rows


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read(path: Path) -> Any:
    envelope = json.loads(path.read_text())
    if (
        set(envelope) != {"sha256", "value"}
        or _digest(envelope["value"]) != envelope["sha256"]
    ):
        raise ValueError(f"invalid integrity seal: {path.name}")
    return envelope["value"]


def _seal(path: Path, value: Any) -> None:
    """Atomically publish a checkpoint; never silently replace different content."""
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"checkpoint content differs: {path.name}")
        return
    encoded = json.dumps({"sha256": _digest(value), "value": value}, allow_nan=False)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _plan(config: ReferenceConfig) -> dict[str, Any]:
    """Bind configuration, implementation, runtime and acquisition order."""
    return {
        "protocol": PROTOCOL,
        "config": asdict(config),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "packet_sha256": PACKET_SHA256,
        "catalog": [asdict(w) for w in catalog()],
        "acquisition": {
            "replicates": REPLICATES,
            "sigma_x": SIGMA_X,
            "sigma_ligand": SIGMA_LIGAND,
            "alpha": ALPHA,
            "time": TIME,
            "z": Z,
            "radius_x": RADIUS_X,
            "radius_ligand": RADIUS_LIGAND,
            "schedule": "PCG64; catalog order; all raw x then raw ligand per world",
        },
    }


def _draw_world(
    rng: np.random.Generator, world: World, config: ReferenceConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Preserve Lyra's exact replicate-generation schedule."""
    shape = (config.trials_per_world, REPLICATES)
    return (
        rng.normal(world.mu_x, SIGMA_X, size=shape),
        rng.normal(world.mu_ligand, SIGMA_LIGAND, size=shape),
    )


def _load(root: Path, config: ReferenceConfig) -> list[dict[str, Any]]:
    """Check sealed, contiguous world checkpoints and the PCG64 state chain."""
    records = []
    expected_state = np.random.Generator(
        np.random.PCG64(config.seed)
    ).bit_generator.state
    expected_shape = (config.trials_per_world, REPLICATES)
    paths = {p.name for p in root.glob("world-*.json")}
    for world_id in range(9):
        path = root / f"world-{world_id}.json"
        if not path.exists():
            break
        value = _read(path)
        required = {"world_id", "plan_sha256", "rng_before", "rng_after", "x", "ligand"}
        if (
            set(value) != required
            or value["world_id"] != world_id
            or value["plan_sha256"] != _digest(_plan(config))
            or value["rng_before"] != expected_state
        ):
            raise ValueError("checkpoint identity or RNG state chain differs")
        for name in ("x", "ligand"):
            array = np.asarray(value[name], dtype=float)
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise ValueError("checkpoint replicate shape or values differ")
        # Setting the state also validates its PCG64 schema without drawing data.
        verifier = np.random.Generator(np.random.PCG64())
        verifier.bit_generator.state = value["rng_after"]
        expected_state = value["rng_after"]
        records.append(value)
    if paths != {f"world-{i}.json" for i in range(len(records))}:
        raise ValueError("noncontiguous or unexpected world checkpoints")
    return records


def _interval(count: int, trials: int) -> list[float]:
    """Descriptive per-world Clopper-Pearson interval; not simultaneous."""
    return [
        0.0 if count == 0 else float(beta.ppf(0.025, count, trials - count + 1)),
        1.0 if count == trials else float(beta.ppf(0.975, count + 1, trials - count)),
    ]


def _summarize(
    config: ReferenceConfig, records: list[dict[str, Any]]
) -> dict[str, Any]:
    per_world = []
    totals = dict.fromkeys(
        score_masks(np.array([], dtype=int), np.array([], dtype=int), 0), 0
    )
    for value in records:
        first, second = classify(
            np.asarray(value["x"]).mean(axis=1),
            np.asarray(value["ligand"]).mean(axis=1),
        )
        metrics = score_masks(first, second, value["world_id"])
        for name, count in metrics.items():
            totals[name] += count
        per_world.append(
            {
                "world_id": value["world_id"],
                "trials": config.trials_per_world,
                "metrics": metrics,
                "ci95_per_metric_not_simultaneous": {
                    name: _interval(count, config.trials_per_world)
                    for name, count in metrics.items()
                },
            }
        )
    if any(project(mask) != diagnosis(mask) for mask in range(512)):
        raise ValueError("predicate projection differs from diagnosis baseline")
    probabilities = analytic_reference()
    return {
        "protocol": PROTOCOL,
        "plan_sha256": _digest(_plan(config)),
        "expected_datasets": 9 * config.trials_per_world,
        "recorded_datasets": len(records) * config.trials_per_world,
        "completed_worlds": list(range(len(records))),
        "missing_worlds": list(range(len(records), 9)),
        "complete": len(records) == 9,
        "totals": totals,
        "per_world": per_world,
        "all_512_projection_masks_agree": True,
        "analytic_reference": probabilities,
        "expected_false_claim_datasets_full_plan": config.trials_per_world
        * sum(p["any_false_claim_probability"] for p in probabilities),
        "scope": "Synthetic fixed catalog, known noise, fixed two-stage acquisition",
    }


def report_control(root: Path) -> dict[str, Any]:
    """Read and validate existing evidence only; never generate missing datasets."""
    plan = _read(root / "plan.json")
    config = ReferenceConfig(**plan["config"])
    if plan != _plan(config):
        raise ValueError("plan source, runtime or protocol differs")
    summary = _summarize(config, _load(root, config))
    if (root / "summary.json").exists() and _read(root / "summary.json") != summary:
        raise ValueError("saved summary differs from checkpoint evidence")
    return summary


def run_control(
    root: Path, config: ReferenceConfig, *, max_new_worlds: int | None = None
) -> dict[str, Any]:
    """Run/resume a locked simulation, atomically checkpointing each whole world.

    An interruption inside a world replays that unfinished world from its saved
    boundary state. Completed worlds are never sampled again. Count/source/runtime
    changes require a fresh output root, including after a fully completed run.
    """
    if not isinstance(config, ReferenceConfig):
        raise ValueError("config must be a ReferenceConfig")
    if max_new_worlds is not None and (
        type(max_new_worlds) is not int or max_new_worlds < 0
    ):
        raise ValueError("max_new_worlds must be a nonnegative integer or None")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _seal(root / "plan.json", _plan(config))
        records = _load(root, config)
        if (root / "summary.json").exists():
            return report_control(root)
        rng = np.random.Generator(np.random.PCG64(config.seed))
        if records:
            rng.bit_generator.state = records[-1]["rng_after"]
        stop = 9 if max_new_worlds is None else min(9, len(records) + max_new_worlds)
        for world in catalog()[len(records) : stop]:
            before = rng.bit_generator.state
            x, ligand = _draw_world(rng, world, config)
            value = {
                "world_id": world.world_id,
                "plan_sha256": _digest(_plan(config)),
                "rng_before": before,
                "rng_after": rng.bit_generator.state,
                "x": x.tolist(),
                "ligand": ligand.tolist(),
            }
            _seal(root / f"world-{world.world_id}.json", value)
            records.append(value)
        summary = _summarize(config, records)
        if summary["complete"]:
            _seal(root / "summary.json", summary)
        return summary
