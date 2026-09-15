"""One explicit warm-start window, with immutable parent and spent-budget ledger.

This is a new least-squares invocation, not restoration of its trust-region
state. No collocation, start portfolio or automatic extension is performed.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from time import time

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.fitter import _training_variables
from autoformalism.fitting.models import FitConfig
from autoformalism.schemas.fit_continuation import (
    ContinuationResult,
    ContinuationSelection,
)
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)


def _verified_parent(directory: Path) -> dict:
    """Read a historical seal without requiring the old Python source version."""
    frozen = public._read(directory / "freeze.json")
    payload = {k: v for k, v in frozen.items() if k != "identity"}
    if public.content_sha256(payload) != frozen["identity"]:
        raise ValueError("parent freeze digest differs")
    request = PublicFitRequest.model_validate(frozen["request"])
    train = PublicSplit.model_validate(frozen["training"])
    val = PublicSplit.model_validate(frozen["validation"])
    rebuilt = public._bundle(request, train, val)
    # Historical code is bound, not substituted. Everything scientific/numerical
    # must reconstruct exactly under this version before continuation is allowed.
    ignored = {"identity", "source_sha256"}
    if {k: v for k, v in rebuilt.items() if k not in ignored} != {
        k: v for k, v in frozen.items() if k not in ignored
    }:
        raise ValueError("parent lowering, data or numerical settings differ")
    envelope = public._read(directory / "result.json")
    if public.content_sha256(envelope["result"]) != envelope["sha256"]:
        raise ValueError("parent result digest differs")
    result = PublicFitResult.model_validate(envelope["result"])
    for key, value in public._result_base(frozen, request).items():
        if public._jsonable(getattr(result, key)) != public._jsonable(value):
            raise ValueError(f"parent result lineage differs: {key}")
    backend = public._read(directory / "backend_result.json")
    if public.content_sha256(backend) != result.backend_result_sha256:
        raise ValueError("parent backend digest differs")
    evidence = public._evidence(request, backend)
    for key, value in evidence.items():
        if public._jsonable(getattr(result, key)) != public._jsonable(value):
            raise ValueError(f"parent public/backend evidence differs: {key}")
    if request.profile != "collocation-feasible-v1":
        raise ValueError("pilot requires the original collocation-feasible-v1 parent")
    if not result.parameters or not result.training.available:
        raise ValueError("parent requires a retained complete finite training fit")
    model, _, _ = public._lower(request)
    variables = _training_variables(model, public.unpack_split(train), FitConfig())
    names = {v.name.removeprefix("parameter:") for v in variables}
    if names != set(result.parameters):
        raise ValueError("parent must retain every equation and initializer parameter")
    if any(
        not v.lower <= result.parameters[v.name.removeprefix("parameter:")] <= v.upper
        for v in variables
    ):
        raise ValueError("parent retained parameter lies outside its domain")
    return {"freeze": frozen, "result": envelope, "backend": backend}


def progress_gate(parent: dict, selection: ContinuationSelection) -> dict:
    """Audit this pilot's training-only progress criterion, not scientific quality."""
    result, raw = parent["result"]["result"], parent["backend"]
    refinement = raw.get("refinement") or {}
    reasons = []
    if refinement.get("budget_exhausted") is not True:
        reasons.append("parent refinement did not exhaust its budget")
    stages = [
        stage["result"]
        for stage in refinement.get("stages", [])
        if stage.get("mode") == "sensitivity"
        and isinstance(stage.get("result"), dict)
        and stage["result"].get("parameters") == result["parameters"]
        and (stage["result"].get("best_evaluated") or {}).get("parameters")
        == result["parameters"]
    ]
    stage = stages[-1] if stages else {}
    if stage.get("optimizer_status") not in (-2, 0):
        reasons.append("selected point is not from a budget-stopped sensitivity stage")
    if stage.get("optimizer_native_success") is not False:
        reasons.append("selected sensitivity stage is not explicitly unconverged")
    history = stage.get("iterations", [])
    window = history[-selection.recent_iterations :]
    costs = [row.get("cost") for row in window]
    valid = len(window) == selection.recent_iterations and all(
        isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in costs
    )
    if valid:
        valid = all(
            a["iteration"] < b["iteration"]
            and a["nfev"] < b["nfev"]
            and a["cost"] >= b["cost"]
            for a, b in pairwise(window)
        )
        valid = valid and window[-1].get("parameters") == result["parameters"]
        valid = valid and stage.get("cost") == costs[-1]
        valid = valid and (stage.get("best_evaluated") or {}).get("cost") == costs[-1]
    drop = (costs[0] - costs[-1]) / max(costs[0], 1e-30) if valid else None
    if drop is None or drop < selection.minimum_relative_cost_drop:
        reasons.append("insufficient coherent recent training-cost improvement")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "training_only": True,
        "recorded_iterations": len(history),
        "recent_costs": costs if valid else [],
        "relative_cost_drop": drop,
        "parent_training_cost": stage.get("cost"),
        "automatic_extension_policy": False,
    }


def _ledger_path(parent_path: Path, parent_identity: str) -> Path:
    # Outside the old experiment root (whose fit child is the parent).
    return parent_path.parent.parent / ".public-fit-continuations" / parent_identity


def _reservation(frozen: dict, directory: Path) -> dict:
    return {
        "parent_identity": frozen["parent"]["freeze"]["identity"],
        "parent_result_sha256": frozen["parent"]["result"]["sha256"],
        "continuation_identity": frozen["identity"],
        "output": str(directory.resolve()),
    }


def _check_ledger(frozen: dict, directory: Path) -> None:
    ledger = Path(frozen["ledger_path"]) / "reservation.json"
    if public._read(ledger) != _reservation(frozen, directory):
        raise ValueError("continuation reservation differs; no fresh budget")


def prepare_continuation(
    parent_path: Path,
    selection: ContinuationSelection,
    directory: Path,
) -> dict:
    """Reserve the only extra window and seal exact parent provenance read-only."""
    parent_path, directory = parent_path.resolve(), directory.resolve()
    experiment = parent_path.parent
    if directory.is_relative_to(experiment) or experiment.is_relative_to(directory):
        raise ValueError("continuation output must be separate from parent experiment")
    parent = _verified_parent(parent_path)
    result = parent["result"]["result"]
    if result["lowered_candidate_sha256"] != selection.parent_lowered_candidate_sha256:
        raise ValueError("selected parent candidate digest differs")
    if result["backend_result_sha256"] != selection.parent_backend_result_sha256:
        raise ValueError("selected parent backend digest differs")
    ledger = _ledger_path(parent_path, parent["freeze"]["identity"])
    payload = {
        "protocol": "public-fit-continuation-freeze-1",
        "selection": selection.model_dump(mode="json"),
        "parent": parent,
        "parent_path": str(parent_path),
        "ledger_path": str(ledger),
        "gate": progress_gate(parent, selection),
        "source_sha256": public._source_identity(),
        "budgets": {
            "original_initializer_seconds": 120,
            "original_refinement_seconds": 180,
            "additional_refinement_seconds": 180,
            "cumulative_refinement_seconds": 360,
            "cumulative_fit_seconds": 480,
            "additional_residual_calls": 240,
            "cumulative_residual_call_cap": 480,
            "setup_and_scoring_in_numerical_budget": False,
        },
    }
    frozen = {**payload, "identity": public.content_sha256(payload)}
    with public._lock(ledger):
        reservation = ledger / "reservation.json"
        expected = _reservation(frozen, directory)
        if reservation.exists() and public._read(reservation) != expected:
            raise ValueError("parent already reserved a continuation; no fresh budget")
        with public._lock(directory):
            path = directory / "freeze.json"
            if path.exists():
                if public._read(path) != frozen:
                    raise ValueError("existing continuation freeze differs")
            else:
                if any(p.name != ".lock" for p in directory.iterdir()):
                    raise ValueError("continuation output is not empty")
                if not reservation.exists():
                    public._write(reservation, expected)
                public._write(path, frozen)
            if not reservation.exists():
                raise ValueError("missing original continuation reservation")
    return inspect_continuation(directory)


def _load(directory: Path) -> dict:
    frozen = public._read(directory / "freeze.json")
    if (
        public.content_sha256({k: v for k, v in frozen.items() if k != "identity"})
        != frozen["identity"]
    ):
        raise ValueError("continuation freeze digest differs")
    if frozen["source_sha256"] != public._source_identity():
        raise ValueError("continuation source changed")
    if _verified_parent(Path(frozen["parent_path"])) != frozen["parent"]:
        raise ValueError("parent artifacts changed")
    selection = ContinuationSelection.model_validate(frozen["selection"])
    if progress_gate(frozen["parent"], selection) != frozen["gate"]:
        raise ValueError("continuation gate evidence differs")
    _check_ledger(frozen, directory)
    return frozen


def inspect_continuation(directory: Path) -> dict:
    """Read-only contract and progress inspection; no numerical fitting."""
    frozen = _load(directory)
    return {
        "identity": frozen["identity"],
        "gate": frozen["gate"],
        "budgets": frozen["budgets"],
        "parent_identity": frozen["parent"]["freeze"]["identity"],
        "parameter_count": len(frozen["parent"]["result"]["result"]["parameters"]),
        "attempt_started": (directory / "started.json").exists(),
        "result_exists": (directory / "result.json").exists(),
    }


def _base_result(frozen: dict) -> dict:
    parent = frozen["parent"]["result"]["result"]
    return {
        "identity": frozen["identity"],
        "parent_identity": parent["identity"],
        "parent_result_sha256": frozen["parent"]["result"]["sha256"],
        "parameters": parent["parameters"],
        "training": parent["training"],
        "validation": parent["validation"],
    }


def _run_extension(frozen: dict, directory: Path) -> dict:
    from autoformalism.fitting.continuation_numerics import run_extension

    return run_extension(frozen, directory)


def execute_continuation(directory: Path) -> ContinuationResult:
    """Run once; interrupted/failed continuations retain the immutable incumbent."""
    with public._lock(directory):
        frozen = _load(directory)
        base = _base_result(frozen)
        path = directory / "result.json"
        if path.exists():
            saved = public._read(path)
            if public.content_sha256(saved["result"]) != saved["sha256"]:
                raise ValueError("continuation result digest differs")
            result = ContinuationResult.model_validate(saved["result"])
            for key in ("identity", "parent_identity", "parent_result_sha256"):
                if getattr(result, key) != base[key]:
                    raise ValueError("continuation result lineage differs")
            if (
                result.backend_result_sha256
                and public.content_sha256(
                    public._read(directory / "backend_result.json")
                )
                != result.backend_result_sha256
            ):
                raise ValueError("continuation backend digest differs")
            return result
        if not frozen["gate"]["eligible"]:
            result = ContinuationResult(
                **base,
                status="ineligible",
                feedback_status="continuation_not_run",
                message="Pilot progress gate not met; parent retained; no new fit.",
            )
        elif (directory / "started.json").exists():
            if (
                public._read(directory / "started.json")["identity"]
                != frozen["identity"]
            ):
                raise ValueError("continuation started identity differs")
            result = ContinuationResult(
                **base,
                status="interrupted",
                feedback_status="numerical_failure_unresolved",
                message=(
                    "Attempt interrupted; checkpoints and parent retained; "
                    "no fresh budget."
                ),
            )
        else:
            public._write(
                directory / "started.json",
                {
                    "identity": frozen["identity"],
                    "utc_seconds": time(),
                    "runtime": public._runtime(),
                },
            )
            try:
                raw = _run_extension(frozen, directory)
                public._write(directory / "backend_result.json", raw)
                result = ContinuationResult(
                    **{**base, **raw["result_fields"]},
                    backend_result_sha256=public.content_sha256(raw),
                )
            except Exception as error:
                result = ContinuationResult(
                    **base,
                    status="extension_failed",
                    feedback_status="numerical_failure_unresolved",
                    message=f"Parent retained; {type(error).__name__}: {error}",
                )
        payload = result.model_dump(mode="json")
        public._write(
            path, {"result": payload, "sha256": public.content_sha256(payload)}
        )
        return result
