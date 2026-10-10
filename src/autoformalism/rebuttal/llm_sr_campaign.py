"""Freeze, run and report a Phase-B LLM-SR campaign.

Mirrors the LLM-ODE campaign: the plan is sealed before any provider call, a
completed task resumes without spending one, and a completed search seals the
model beside its result so the frozen evaluator can adapt it.

LLM-SR learns one function per run, so a cell with several targets runs their
pipeline once per target, which is what their specification format requires.
Phase C cells enter through ``prepare_phase_c`` and ``run_phase_c``, which read
a verified release and resume only against the endpoint kind frozen in the plan.
A Phase C task's targets can be searched by separate processes at once; the
task's model is assembled and sealed once all of them have finished.

The model kept is LLM-SR's own choice: the sample its evaluator scored highest.
The development rollout error is reported beside it and chooses nothing.

A Phase C plan may declare that LLM-SR takes the regression table the PySR
baseline fits, since both fit derivative labels; current plans keep its rows
in time order. A plan that does may also declare a program rollout: a selected
program that reads the history is then sealed as the program it is, with its
coefficients and its grid, and rolled out by ``llm_sr_programs``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np

from autoformalism.baselines.core import numeric_feature_names, regression_table
from autoformalism.baselines.models import BaselineDevelopmentResult
from autoformalism.data import DatasetSplit
from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.baseline_validation import load_public
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate
from autoformalism.rebuttal.llm_ode_campaign import (
    cell_arrays,
    observed_channels,
    phase_c_rows,
    public_prompt_text,
    public_task_specification,
)
from autoformalism.rebuttal.llm_sr_driver import build_searcher  # noqa: F401
from autoformalism.rebuttal.llm_sr_programs import EXECUTION as PROGRAM_EXECUTION
from autoformalism.rebuttal.phase_c_baselines import PhaseCBaselineCell
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    endpoint_environment,
    load_development,
    load_phase_c_vendored_plan,
    require_endpoint,
    task_lock,
)
from autoformalism.rebuttal.prefit_replay import (
    content_hash,
    sealed_read,
    sealed_write,
)
from autoformalism.rebuttal.vendored_campaign import VendoredCampaignPlan

PROTOCOL = "phase-b-llm-sr-campaign-1"
PHASE_C_PROTOCOL = "phase-c-llm-sr-campaign-1"

#: LLM-SR's own rule: of every sample its evaluator scored, the one with the
#: highest score, the negative mean squared error of the program fitted to the
#: training derivatives.
SELECTION = "highest_llm_sr_training_score"


class SearcherFactory(Protocol):
    """Runs upstream's search for one cell, injected so tests need no provider."""

    def __call__(self, **kwargs) -> dict: ...


def pysr_features(cell: PhaseCBaselineCell) -> tuple[str, ...]:
    """The channels PySR regresses on in a cell: numeric ones, targets first."""
    return numeric_feature_names(cell.dataset.train, cell.context)


def training_rows(
    plan: dict, row: dict, train: DatasetSplit, context: ValidationContext
) -> tuple[np.ndarray, np.ndarray]:
    """The rows LLM-SR searches: one per training point, and its labels.

    A plan that declares PySR's regression table gets PySR's rows, channels
    and numpy.gradient labels, in time order or in the order its seed fixes.
    Otherwise the cell's arrays are stacked in time order with fourth-order
    derivatives, as Phase B gave them. Labels have one column per searched
    target.
    """
    channels = tuple(row["channels"])
    targets = tuple(context.targets)
    table = plan.get("regression_table")
    if table is None:
        arrays = cell_arrays(train)
        if tuple(arrays.channels) != channels:
            raise ValueError("public development input drift: channels differ")
        columns = [channels.index(target) for target in targets]
        return arrays.states, arrays.derivatives[:, columns]
    names = numeric_feature_names(train, context)
    if names != channels:
        raise ValueError(
            "public development input drift: PySR's features differ from the "
            "frozen channels"
        )
    values, labels, _ = regression_table(train, names, targets)
    if table["row_order"] == "time":
        return values, labels
    order = np.random.default_rng(table["row_order_seed"]).permutation(len(values))
    return values[order], labels[order]


def environment_identity() -> dict:
    """Bind resume to the code and endpoint kind, never to a per-job port.

    Phase B campaigns always ran against a job-local vLLM.
    """
    return endpoint_environment("job_local_vllm")


def prepare(config_path: Path, public_root: Path, root: Path) -> dict:
    """Freeze the campaign before any provider call."""
    plan = VendoredCampaignPlan.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    if plan.method != "llm_sr":
        raise ValueError(f"expected an llm_sr plan, not {plan.method!r}")
    public = public_root.expanduser().resolve()
    rows = []
    for cell in plan.cells:
        development, context, identity = load_public(
            public, cell.benchmark_id, cell.tier
        )
        channels = observed_channels(development.train)
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
            # One request per samples_per_prompt samples, per searched target.
            "maximum_logical_samples": sum(
                plan.budget.declared * len(row["searched_targets"]) for row in rows
            ),
            "rows": rows,
        },
    )


def prepare_phase_c(config_path: Path, release: Path, root: Path) -> dict:
    """Freeze a Phase C campaign against one verified release, before any call."""
    plan = load_phase_c_vendored_plan(config_path)
    if plan.method != "llm_sr":
        raise ValueError(f"expected an llm_sr plan, not {plan.method!r}")
    release = release.expanduser().resolve()
    receipt, rows = phase_c_rows(
        plan,
        release,
        features=None if plan.regression_table is None else pysr_features,
    )
    root.mkdir(parents=True, exist_ok=True)
    return sealed_write(
        root / "plan.json",
        {
            "protocol": PHASE_C_PROTOCOL,
            "plan": plan.model_dump(mode="json"),
            "release": str(release),
            "release_summary_sha256": receipt,
            "environment": endpoint_environment(plan.endpoint),
            "reporting_qualifications": list(plan.reporting_qualifications()),
            # One request per samples_per_prompt samples, per searched target.
            "maximum_logical_samples": sum(
                plan.budget.declared * len(row["searched_targets"]) for row in rows
            ),
            "rows": rows,
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )


def run(root: Path, index: int, *, search: SearcherFactory | None = None) -> dict:
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
    row, directory, finished = _task(sealed, root, index)
    if finished is not None:
        return finished
    development, context, identity = load_public(
        Path(sealed["public_root"]), row["benchmark_id"], row["tier"]
    )
    if identity != row["public_identity"]:
        raise ValueError("public development input drift")
    return _search_and_seal(sealed, row, directory, development, context, search)


def run_phase_c(
    root: Path,
    index: int,
    *,
    endpoint: str,
    search: SearcherFactory | None = None,
    target: str | None = None,
) -> dict:
    """Resume one Phase C task, only against the endpoint kind it was frozen for.

    With `target`, only that target is searched and kept for the task, and
    nothing is sealed. Without it, every target not yet searched is searched in
    turn, then the task's model is assembled and sealed.
    """
    sealed = sealed_read(root / "plan.json")
    if sealed["protocol"] != PHASE_C_PROTOCOL:
        raise ValueError(f"{root / 'plan.json'} is not a Phase C LLM-SR plan")
    require_endpoint(sealed, root, endpoint)
    if not 0 <= index < len(sealed["rows"]):
        raise ValueError("task index out of range")
    targets = tuple(sealed["rows"][index]["searched_targets"])
    if target is not None and target not in targets:
        raise ValueError(f"task {index} searches {', '.join(targets)}, not {target!r}")
    held = targets if target is None else (target,)
    with _held(root, index, held, whole_task=target is None):
        row, directory, finished = _task(sealed, root, index)
        if finished is not None:
            return finished
        development, context = load_development(sealed, row)
        return _search_and_seal(
            sealed, row, directory, development, context, search, only=target
        )


@contextmanager
def _held(
    root: Path, index: int, targets: tuple[str, ...], *, whole_task: bool
) -> Iterator[None]:
    """Hold the targets this process searches, and the task when it seals it.

    A target search cannot resume, so the driver sets a stopped one aside
    before searching that target again; holding the target's lock makes that
    safe. The task directory is never set aside, because it keeps the targets
    that finished.
    """
    with ExitStack() as stack:
        if whole_task:
            stack.enter_context(task_lock(root, str(index)))
        for name in targets:
            stack.enter_context(task_lock(root, f"{index}-{name}"))
        yield


def _task(sealed: dict, root: Path, index: int) -> tuple[dict, Path, dict | None]:
    """Locate one task, and its sealed result when it already finished."""
    if not 0 <= index < len(sealed["rows"]):
        raise ValueError("task index out of range")
    row = sealed["rows"][index]
    directory = root / "results" / str(index)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    return row, directory, sealed_read(result_path) if result_path.exists() else None


def _search_and_seal(
    sealed: dict,
    row: dict,
    directory: Path,
    development,
    context,
    search: SearcherFactory | None,
    *,
    only: str | None = None,
) -> dict:
    """Run upstream's search on development data and seal what it selected.

    With `only`, that one target is searched and kept, and nothing is sealed.
    """
    result_path = directory / "result.json"
    if search is None:  # pragma: no cover - requires the vendored checkout
        raise ValueError(
            "supply a searcher factory bound to the pinned LLM-SR checkout"
        )
    from autoformalism.rebuttal.llm_ode_driver import development_rollout_error

    values, labels = training_rows(sealed["plan"], row, development.train, context)
    outcome = search(
        channels=tuple(row["channels"]),
        targets=tuple(context.targets),
        values=values,
        labels=labels,
        description=row.get("prompt", ""),
        directory=directory,
        development=(development.train, development.validation),
        context=context,
        score_rollout=development_rollout_error,
        **({} if only is None else {"only": only}),
    )
    if only is not None:
        return outcome

    if outcome["status"] == "complete":
        programs = outcome.get("programs")
        if programs:
            # A program that reads the history has no equation; the shared
            # evaluator cannot run it, so the selection names how it is run.
            equations: dict[str, str] = {}
            payload: dict[str, object] = {
                "execution": PROGRAM_EXECUTION,
                "programs": programs,
                "channels": list(row["channels"]),
                "grid_step": outcome["grid_step"],
            }
        else:
            equations = outcome["equations"]
            payload = {
                "candidate": equation_candidate(
                    "llm_sr", equations, context
                ).model_dump(mode="json"),
                # Coefficients are refitted into the expressions, so none remain.
                "parameters": {},
            }
        selection = BaselineDevelopmentResult(
            method="llm_sr",
            benchmark_id=row["benchmark_id"],
            tier=row["tier"],
            seed=row["repetition"],
            equations=equations,
            selected_hyperparameters={
                "llm_samples": int(sealed["plan"]["budget"]["declared"]),
                "num_islands": int(sealed["plan"].get("islands", 10)),
                "selection": SELECTION,
            },
            selection_payload=payload,
            training_normalized_mse=float(outcome["training_rollout_error"]),
            validation_normalized_mse=float(outcome["development_rollout_error"]),
            elapsed_wall_seconds=outcome.get("accounting", {}).get(
                "wall_seconds", outcome.get("accounting", {}).get("search_seconds")
            ),
        )
        sealed_write(
            directory / "native-selection.json",
            {
                "plan_sha256": sealed["artifact_sha256"],
                "selection": selection.model_dump(mode="json"),
            },
        )
    return sealed_write(
        result_path,
        {
            **{
                key: row[key]
                for key in ("index", "benchmark_id", "tier", "repetition")
            },
            "protocol": sealed["protocol"],
            "plan_sha256": sealed["artifact_sha256"],
            "status": outcome["status"],
            "error": outcome.get("error"),
            "equations": outcome.get("equations"),
            "programs": outcome.get("programs"),
            "grid_step": outcome.get("grid_step"),
            "selected_samples": outcome.get("selected_samples"),
            "development_rollout_error": outcome.get("development_rollout_error"),
            "accounting": outcome.get("accounting", {}),
            "test_data_opened": False,
            "private_reference_opened": False,
            "selection_metric": SELECTION,
        },
    )


def report(root: Path) -> dict:
    """Coverage before scores, and the same persisted summary D3 writes."""
    sealed = sealed_read(root / "plan.json")
    rows = []
    for task in sealed["rows"]:
        path = root / "results" / str(task["index"]) / "result.json"
        rows.append(
            sealed_read(path) if path.exists() else {**task, "status": "pending"}
        )
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    expected = len(sealed["rows"])
    complete = counts.get("complete", 0)
    value = {
        # The plan's own protocol, so one report serves Phase B and Phase C.
        "protocol": sealed["protocol"],
        "status": "complete" if complete == expected else "pending",
        "expected": expected,
        "terminal_success": complete,
        "not_started": counts.get("pending", 0),
        "terminal_scientific_failure": sum(
            count
            for status, count in counts.items()
            if status in {"inexpressible", "rollout_failed", "no_candidates"}
        ),
        "infrastructure_failure": counts.get("endpoint_unavailable", 0),
        "frozen_models": sum(
            1
            for task in sealed["rows"]
            if (
                root / "results" / str(task["index"]) / "native-selection.json"
            ).is_file()
        ),
        # Tasks whose model is a program that reads the history, and of their
        # targets, those whose output changed when later rows were removed.
        "program_models": sum(1 for item in rows if item.get("programs")),
        "programs_reading_later_rows": sum(
            1
            for item in rows
            for record in (item.get("programs") or {}).values()
            if record["look_ahead"]["reads_later_rows"]
        ),
        "counts": counts,
        "reporting_qualifications": sealed["reporting_qualifications"],
        "rows": rows,
    }
    value["artifact_sha256"] = content_hash(value)
    atomic_json(root / "summary.json", value)
    return value
