"""Run LLM-ODE's search on Phase-B cells at a pinned upstream revision.

The search is not reimplemented. Upstream's per-variable searcher receives our
trajectories, and upstream's own `System` evaluates the assembled candidates.
This module supplies data, applies the declared prompt adaptation, reproduces
upstream's selection rule, and seals the result.

Three accommodations are forced by the benchmark and are declared in the plan
rather than hidden here:

* Upstream's driver builds its data by integrating a ground-truth equation it
  is given. We cannot do that, so the per-variable searcher is constructed
  directly from our observed trajectories.
* Upstream assumes one trajectory per problem. A Phase-B cell has several, so
  derivatives are estimated within each trajectory and the rows stacked. The
  regression is pointwise, so stacking is sound; differentiating across a
  join would not be.
* Upstream's selection hands its evaluator the sealed test trajectory. Ours
  cannot see it, so selection uses the same rule -- product of the per-variable
  Pareto frontiers, ranked by rolled-out error -- over development data only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from autoformalism.data import BenchmarkRegistry, DatasetSplit
from autoformalism.rebuttal.baseline_validation import load_public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.rebuttal.vendored_campaign import VendoredCampaignPlan

#: Upstream differentiates with findiff at fourth order; keep that choice.
DERIVATIVE_ACCURACY = 4


class SearcherFactory(Protocol):
    """Runs upstream's search for one cell, injected so tests need no provider.

    Upstream's own ``LlmOde`` wrapper builds its data by integrating the true
    system, so it cannot be handed observed trajectories. The driver therefore
    drives ``LlmOdeEquation`` -- the per-variable searcher ``LlmOde`` itself
    composes -- once per target, which is why the cell is the unit here.
    """

    def __call__(
        self,
        *,
        train: CellArrays,
        validation: CellArrays,
        targets: tuple[str, ...],
        prompt: str,
        directory: Path,
        development: tuple[DatasetSplit, DatasetSplit],
        context: object,
    ) -> dict: ...


@dataclass(frozen=True)
class CellArrays:
    """Development data in the shape upstream's searcher expects."""

    time: NDArray[np.float64]
    states: NDArray[np.float64]
    derivatives: NDArray[np.float64]
    channels: tuple[str, ...]
    trajectory_bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.states.shape != self.derivatives.shape:
            raise ValueError("states and derivatives must have the same shape")
        if self.states.shape[1] != len(self.channels):
            raise ValueError("state columns must match the observed channels")
        if not self.trajectory_bounds:
            raise ValueError("require at least one trajectory")


def observed_channels(split: DatasetSplit) -> tuple[str, ...]:
    """Name the channels upstream will treat as system variables."""
    first = split.trajectories[0]
    channels = (*first.targets, *first.auxiliaries)
    for trajectory in split.trajectories:
        if (*trajectory.targets, *trajectory.auxiliaries) != channels:
            raise ValueError("channel identities differ across trajectories")
    return channels


def finite_difference(
    values: NDArray[np.float64], step: float
) -> NDArray[np.float64]:
    """Fourth-order central differences, as upstream's findiff call computes.

    Implemented directly so the campaign does not depend on findiff being
    installed to prepare data; the coefficients are the standard ones upstream
    requests with ``acc=4`` and are checked against an exact polynomial.
    """
    if values.shape[0] < 5:
        raise ValueError("fourth-order differences need at least five samples")
    result = np.empty_like(values, dtype=float)
    interior = slice(2, -2)
    result[interior] = (
        values[:-4] - 8.0 * values[1:-3] + 8.0 * values[3:-1] - values[4:]
    ) / (12.0 * step)
    # One-sided fourth-order stencils keep the endpoints at the same order.
    forward = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / (12.0 * step)
    backward = -forward[::-1]
    count = values.shape[0]
    for index in (0, 1):
        result[index] = np.tensordot(forward, values[index : index + 5], axes=1)
    for index in (count - 2, count - 1):
        result[index] = np.tensordot(backward, values[index - 4 : index + 1], axes=1)
    return result


def cell_arrays(split: DatasetSplit) -> CellArrays:
    """Stack every trajectory after differentiating within each one."""
    channels = observed_channels(split)
    times: list[NDArray[np.float64]] = []
    states: list[NDArray[np.float64]] = []
    derivatives: list[NDArray[np.float64]] = []
    bounds: list[tuple[int, int]] = []
    start = 0
    for trajectory in split.trajectories:
        measured = {**trajectory.targets, **trajectory.auxiliaries}
        block = np.column_stack(
            [np.asarray(measured[name], float) for name in channels]
        )
        time = np.asarray(trajectory.time, dtype=float)
        steps = np.diff(time)
        if not np.allclose(steps, steps[0]):
            raise ValueError("fourth-order differences require a uniform grid")
        derivatives.append(finite_difference(block, float(steps[0])))
        states.append(block)
        times.append(time)
        bounds.append((start, start + len(time)))
        start += len(time)
    return CellArrays(
        time=np.concatenate(times),
        states=np.vstack(states),
        derivatives=np.vstack(derivatives),
        channels=channels,
        trajectory_bounds=tuple(bounds),
    )


def specification_block(
    prompt: str, channels: tuple[str, ...], variable_index: int
) -> str:
    """Append the public task specification to an upstream prompt.

    A declared adaptation: upstream withholds variable meanings to mitigate
    memorisation, which this benchmark controls for with its named and
    obfuscated pairing. The wording is fixed and adds no private information,
    no reference knowledge and nothing about our own pipeline.
    """
    named = ", ".join(f"x_{index} = {name}" for index, name in enumerate(channels))
    return (
        "\n\nTask specification (public):\n"
        f"{prompt.strip()}\n\n"
        f"Variables: {named}.\n"
        f"You are proposing the time derivative of x_{variable_index} "
        f"({channels[variable_index]})."
    )


#: The public prompt is rendered in lettered sections. A and B are the public
#: scientific content -- the task and the channels -- and are what the other
#: LLM baselines receive. C onward is this pipeline's modelling requirements
#: and response format, which must not enter a vendored method's prompt.
SPECIFICATION_SECTIONS = ("A. Task specification", "B. Available data")
SPECIFICATION_END = "C. Modeling requirements"


def public_prompt_text(root: Path, benchmark: str, tier: str) -> str:
    """Read the rendered public proposer prompt for one cell."""
    spec = BenchmarkRegistry().get(benchmark)
    prompt_root = root / spec.relative_root
    if spec.data_layout != "tidy_split_file":
        prompt_root /= spec.tier_directory_template.format(tier=tier)
    return (prompt_root / "proposer_prompt.txt").read_text(encoding="utf-8")


def public_task_specification(prompt: str) -> str:
    """Take the public scientific sections of the proposer prompt, and no more.

    Pasting the whole prompt would hand a vendored baseline our modelling
    requirements, our response format and our restrictions. Taking a prefix by
    position would silently pass everything if the prompt were ever reordered,
    so each boundary is required to be present.
    """
    for heading in (*SPECIFICATION_SECTIONS, SPECIFICATION_END):
        if heading not in prompt:
            raise ValueError(
                f"public prompt has no {heading!r} section; the specification "
                "boundary must be re-established before it can be supplied"
            )
    return prompt[: prompt.index(SPECIFICATION_END)].strip()


def select_system(
    frontiers: tuple[tuple[str, ...], ...],
    score: Callable[[tuple[str, ...]], float | None],
    *,
    maximum_combinations: int = 10_000,
) -> tuple[str, ...] | None:
    """Rank the product of per-variable frontiers, as upstream's driver does.

    Upstream sorts assembled systems by their rolled-out error and keeps the
    best. The cap exists because the product grows multiplicatively and a
    truncated search would otherwise be reported as a completed one.
    """
    if not frontiers or any(not item for item in frontiers):
        return None
    total = 1
    for item in frontiers:
        total *= len(item)
    if total > maximum_combinations:
        raise ValueError(
            f"{total} assembled systems exceed the {maximum_combinations} cap; "
            "record the truncation rather than silently ranking a subset"
        )
    best: tuple[str, ...] | None = None
    best_score = np.inf
    for combination in product(*frontiers):
        value = score(combination)
        if value is None or not np.isfinite(value):
            continue
        if value < best_score:
            best_score, best = float(value), combination
    return best


PROTOCOL = "phase-b-llm-ode-campaign-1"


def environment_identity() -> dict:
    """Bind resume to the code and endpoint kind, never to a per-job port."""
    return {
        "runtime_source_sha256": runtime_source_hash(),
        "provider": "vllm",
        "endpoint_kind": "job-local vllm endpoint",
    }


def target_indices(
    channels: tuple[str, ...], targets: tuple[str, ...]
) -> tuple[int, ...]:
    """Search only the predicted channels.

    Upstream runs one searcher per state because its systems are autonomous.
    Here the auxiliaries are supplied over the horizon, so only the targets are
    predicted; searching the rest would spend calls on equations the evaluation
    never uses. A declared accommodation, and it reduces the call count from
    iterations x islands x states to iterations x islands x targets.
    """
    missing = [name for name in targets if name not in channels]
    if missing:
        raise ValueError(f"targets absent from the observed channels: {missing}")
    return tuple(channels.index(name) for name in targets)


def prepare(config_path: Path, public_root: Path, root: Path) -> dict:
    """Freeze the campaign before any provider call."""
    plan = VendoredCampaignPlan.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    if plan.method != "llm_ode":
        raise ValueError(f"expected an llm_ode plan, not {plan.method!r}")
    public = public_root.expanduser().resolve()
    rows = []
    for cell in plan.cells:
        development, context, identity = load_public(
            public, cell.benchmark_id, cell.tier
        )
        channels = observed_channels(development.train)
        # Frozen at prepare time so the text a task used is part of the sealed
        # plan rather than something re-read at run time.
        specification = (
            public_task_specification(
                public_prompt_text(public, cell.benchmark_id, cell.tier)
            )
            if plan.prompt_policy.supplies_public_task_specification
            else ""
        )
        for repetition in plan.repetitions:
            rows.append(
                {
                    "index": len(rows),
                    "benchmark_id": cell.benchmark_id,
                    "tier": cell.tier,
                    "repetition": repetition,
                    "public_identity": identity,
                    "channels": list(channels),
                    "searched_targets": list(context.targets),
                    "prompt": specification,
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        root / "plan.json",
        {
            "protocol": PROTOCOL,
            "plan": plan.model_dump(mode="json"),
            "public_root": str(public),
            "environment": environment_identity(),
            "reporting_qualifications": list(plan.reporting_qualifications()),
            "maximum_logical_calls": (
                len(rows) * plan.budget.declared * 4 * max(
                    len(row["searched_targets"]) for row in rows
                )
            ),
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )


def run(
    root: Path,
    index: int,
    *,
    search: SearcherFactory | None = None,
) -> dict:
    """Resume one task; a completed result causes no call and no refitting."""
    sealed = sealed_read(root / "plan.json")
    if (
        sealed["protocol"] != PROTOCOL
        or sealed["environment"] != environment_identity()
    ):
        raise ValueError(
            "protocol or code changed since this plan was frozen. Run from the "
            f"checkout that froze it, or delete {root / 'plan.json'} and its "
            "results to re-freeze at the current code."
        )
    if not 0 <= index < len(sealed["rows"]):
        raise ValueError("task index out of range")
    row = sealed["rows"][index]
    directory = root / "results" / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return sealed_read(result_path)

    development, context, identity = load_public(
        Path(sealed["public_root"]), row["benchmark_id"], row["tier"]
    )
    if identity != row["public_identity"]:
        raise ValueError("public development input drift")
    if search is None:  # pragma: no cover - requires the vendored checkout
        raise ValueError(
            "supply a searcher factory bound to the pinned LLM-ODE checkout"
        )
    train = cell_arrays(development.train)
    validation = cell_arrays(development.validation)
    outcome = search(
        train=train,
        validation=validation,
        targets=tuple(context.targets),
        prompt=row.get("prompt", ""),
        directory=directory,
        # Selection rolls out on development data; the searcher never receives
        # a split it could confuse with the held-out one.
        development=(development.train, development.validation),
        context=context,
    )
    return sealed_write(
        result_path,
        {
            **{
                key: row[key]
                for key in ("index", "benchmark_id", "tier", "repetition")
            },
            "protocol": PROTOCOL,
            "plan_sha256": sealed["artifact_sha256"],
            "status": outcome["status"],
            "error": outcome.get("error"),
            "accounting": outcome.get("accounting", {}),
            "test_data_opened": False,
            "private_reference_opened": False,
            "selection_metric": "development_rollout_error",
        },
    )


def report(root: Path) -> dict:
    """Keep missing and failed cells visible rather than averaging over them."""
    sealed = sealed_read(root / "plan.json")
    rows = []
    for task in sealed["rows"]:
        path = root / "results" / str(task["index"]) / "result.json"
        rows.append(
            sealed_read(path)
            if path.exists()
            else {**task, "status": "pending"}
        )
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "protocol": PROTOCOL,
        "expected": len(sealed["rows"]),
        "counts": counts,
        "reporting_qualifications": sealed["reporting_qualifications"],
        "rows": rows,
    }
