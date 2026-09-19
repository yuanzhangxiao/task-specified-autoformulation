"""Recover fixed-model scoring in a new pilot root, preserving every source result.

No proposal or optimizer is rerun. Only the known empty-vector execution failure
is eligible; existing numerical failures and successful fits remain untouched.
"""

from __future__ import annotations

import copy
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.review_continuation import _check_result
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicFitResult
from autoformalism.staged_topology import content_hash

POLICY = "shared-process-fixed-model-recovery-1"
ERROR = (
    "ValueError: zero-size array to reduction operation maximum which has no identity"
)
KEY = "parameter_free_recovery"


def _eligible(result: dict) -> bool:
    """Require the exact execution defect and no learnable lowered parameters."""
    trial = result.get("trial")
    if result["status"] != "fit_failed" or trial is None:
        return False
    if trial["fit"].get("message") != ERROR:
        return False
    request = PublicFitRequest.model_validate(trial["request"])
    model, _, _ = public._lower(request)
    if model.parameter_names or result.get("selected") is not None:
        raise ValueError("empty-vector failure does not describe a fixed model")
    return True


def _copy(source: Path, dest: Path) -> None:
    data = source.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.read_bytes() != data:
            raise ValueError(f"partial recovery import differs: {dest.name}")
    else:
        temporary = dest.with_suffix(dest.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(dest)


def prepare(source: Path, root: Path) -> dict:
    """Seal a new execution identity and original costs before any evaluation."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and recovery roots must be disjoint")
    with io.execution_lease(source, exclusive=True), public._lock(root):
        io.require_open(source)
        io.require_open(root)
        original = io.verify(source, execution=False)
        if original["protocol"] != io.SHARED_PROTOCOL or KEY in original:
            raise ValueError("requires the original shared-process pilot")
        if (source / "submission-round-1.json").exists() or (
            source / "submission-intent/round-1"
        ).exists():
            raise ValueError(
                "revision round already submitted; inspect before recovery"
            )
        entries = {}
        for task in original["tasks"]:
            directory = io.round_path(source, task, 0)
            if io.round_path(source, task, 1).exists():
                raise ValueError("source revision work exists; inspect before recovery")
            result = io.read_round(source, task, 0)
            if result is None:
                raise ValueError(f"source round incomplete: {task['task_id']}")
            _check_result(result, task, 0)
            proposal = sealed_read(directory / "proposal.json")
            if result["proposal_sha256"] != proposal["artifact_sha256"]:
                raise ValueError("source proposal/result mismatch")
            records = _cache_records(
                directory / "calls",
                content_hash([original["artifact_sha256"], task, 0]),
            )
            cost = _cost(records)
            if result["cost"] != cost:
                raise ValueError("source call accounting differs")
            started = directory / "fit/started.json"
            if started.exists():
                frozen = public._read(directory / "fit/freeze.json")
                if public._read(started)["identity"] != frozen["identity"]:
                    raise ValueError("historical fit marker differs")
            eligible = _eligible(result)
            if eligible:
                if not started.exists():
                    raise ValueError("eligible failed fit lacks its execution marker")
                trial = result["trial"]
                cell = original["cells"][task["cell"]]
                saved = public._read(directory / "fit/result.json")
                if (
                    saved["result"] != trial["fit"]
                    or public.content_sha256(saved["result"]) != saved["sha256"]
                    or public.content_sha256(saved) != trial["fit_result_sha256"]
                    or trial["fit"]["identity"] != frozen["identity"]
                    or frozen["request"] != trial["request"]
                    or frozen["training"] != cell["training"]
                    or frozen["validation"] != cell["validation"]
                    or frozen["identity"]
                    != public.content_sha256(
                        {k: v for k, v in frozen.items() if k != "identity"}
                    )
                ):
                    raise ValueError(
                        "failed fit does not match the sealed public handoff"
                    )
            entries[task["task_id"]] = {
                "result_sha256": result["artifact_sha256"],
                "proposal_sha256": proposal["artifact_sha256"],
                "calls_sha256": content_hash(records),
                "cost": cost,
                "historical_fit_attempts": int(started.exists()),
                "eligible": eligible,
            }
            imported = root / "imports" / task["task_id"]
            _copy(directory / "result.json", imported / "result.json")
            _copy(directory / "proposal.json", imported / "proposal.json")
            sealed_write(imported / "calls.json", {"records": records})
        ledger = {
            "protocol": POLICY,
            "source_root": str(source),
            "source_plan_sha256": original["artifact_sha256"],
            "entries": entries,
            "optimizer_calls": 0,
            "llm_calls": 0,
            "equations_changed": False,
            "automatic_next_round": False,
        }
        if not any(e["eligible"] for e in entries.values()):
            raise ValueError("no qualifying fixed-model failures")
        _copy(source / "plan.json", root / "imports/plan.json")
        for cell, value in original["cells"].items():
            for name in value["assets"]:
                _copy(
                    source / "public/phase_b_v1" / cell / name,
                    root / "public/phase_b_v1" / cell / name,
                )
        plan = {k: v for k, v in original.items() if k != "artifact_sha256"}
        plan.update(
            source_sha256=public._source_identity(),
            runtime=public._runtime(),
            launcher_sha256=io.launcher_hash(io.SHARED_PROTOCOL),
            **{KEY: ledger},
        )
        value = sealed_write(root / "plan.json", plan)
        verify_imports(root, value)
        return value


def _result(original: dict, ledger_entry: dict, trial: dict | None) -> dict:
    """Preserve historical terminal states unless this exact defect was evaluated."""
    value = {k: v for k, v in original.items() if k != "artifact_sha256"}
    value["source_result_sha256"] = original["artifact_sha256"]
    value["execution_recovery"] = POLICY if ledger_entry["eligible"] else "preserved"
    if ledger_entry["eligible"]:
        retained = (
            trial
            if pipeline.selection_key(trial) < pipeline.selection_key(None)
            else None
        )
        value.update(
            trial=trial,
            selected=retained,
            closed=retained is None,
            status=trial["fit"]["status"],
            selected_new_trial=retained is not None,
        )
    return value


def verify_imports(root: Path, plan: dict) -> None:
    """Validate source snapshots and published anchors; partial recovery is explicit."""
    ledger = plan[KEY]
    original = sealed_read(root / "imports/plan.json")
    if (
        ledger["protocol"] != POLICY
        or original["artifact_sha256"] != ledger["source_plan_sha256"]
    ):
        raise ValueError("recovery source identity differs")
    if any(
        plan.get(k) != original.get(k)
        for k in (
            "config",
            "tasks",
            "cells",
            "selection",
            "shared_process_experiment",
        )
    ):
        raise ValueError("recovery changed the scientific experiment")
    if set(ledger["entries"]) != {t["task_id"] for t in plan["tasks"]}:
        raise ValueError("recovery ledger omits a task")
    for task in plan["tasks"]:
        entry = ledger["entries"][task["task_id"]]
        imported = root / "imports" / task["task_id"]
        source = sealed_read(imported / "result.json")
        proposal = sealed_read(imported / "proposal.json")
        records = sealed_read(imported / "calls.json")["records"]
        if (
            source["artifact_sha256"] != entry["result_sha256"]
            or proposal["artifact_sha256"] != entry["proposal_sha256"]
            or content_hash(records) != entry["calls_sha256"]
            or _cost(records) != entry["cost"]
            or _eligible(source) != entry["eligible"]
        ):
            raise ValueError("recovery import or accounting differs")
        result = io.read_round(root, task, 0)
        if result is None:
            continue
        trial = result.get("trial")
        if entry["eligible"]:
            fit_dir = io.round_path(root, task, 0) / "fit"
            saved = public._read(fit_dir / "result.json")
            if public.content_sha256(saved["result"]) != saved["sha256"]:
                raise ValueError("recovered fit digest differs")
            fit = PublicFitResult.model_validate(saved["result"])
            frozen = public._read(fit_dir / "freeze.json")
            cell = plan["cells"][task["cell"]]
            request = PublicFitRequest.model_validate(source["trial"]["request"])
            if (
                frozen["request"] != source["trial"]["request"]
                or frozen["training"] != cell["training"]
                or frozen["validation"] != cell["validation"]
                or frozen["identity"]
                != public.content_sha256(
                    {k: v for k, v in frozen.items() if k != "identity"}
                )
                or any(
                    public._jsonable(getattr(fit, k)) != public._jsonable(v)
                    for k, v in public._result_base(frozen, request).items()
                )
            ):
                raise ValueError("recovered fit lineage differs")
            if fit.backend_result_sha256 is not None:
                raw = public._read(fit_dir / "backend_result.json")
                if (
                    public.content_sha256(raw) != fit.backend_result_sha256
                    or raw.get("execution_mode") != "parameter-free-rollout-1"
                    or raw.get("optimizer_calls") != 0
                ):
                    raise ValueError("recovered backend evidence differs")
            if fit.parameters not in (None, {}) or fit.actual_residual_calls not in (
                None,
                0,
            ):
                raise ValueError("recovery must not fit parameters")
            expected_trial = copy.deepcopy(source["trial"])
            expected_trial.update(
                fit=fit.model_dump(mode="json"),
                fit_result_sha256=public.content_sha256(saved),
                packet=(
                    sealed_read(fit_dir.parent / "packet.json")["packet"]
                    if (fit_dir.parent / "packet.json").exists()
                    else None
                ),
            )
            if trial != expected_trial:
                raise ValueError("recovery changed the saved model or its evidence")
        expected = _result(source, entry, trial)
        if {k: v for k, v in result.items() if k != "artifact_sha256"} != expected:
            raise ValueError("recovered round differs from its source/rollout")


def recover(source: Path, root: Path) -> dict:
    """Evaluate only eligible fixed models; preserve successes and consumed markers."""
    prepare(source, root)
    with io.execution_lease(root), public._lock(root / "recovery-worker"):
        plan = io.verify(root)
        for task in plan["tasks"]:
            entry = plan[KEY]["entries"][task["task_id"]]
            directory = io.round_path(root, task, 0)
            old = sealed_read(root / "imports" / task["task_id"] / "result.json")
            _copy(
                root / "imports" / task["task_id"] / "proposal.json",
                directory / "proposal.json",
            )
            if io.read_round(root, task, 0) is not None:
                continue
            trial = old.get("trial")
            if entry["eligible"]:
                trial = copy.deepcopy(trial)
                request = PublicFitRequest.model_validate(trial["request"])
                cell = plan["cells"][task["cell"]]
                training = public.PublicSplit.model_validate(cell["training"])
                validation = public.PublicSplit.model_validate(cell["validation"])
                public.prepare_fit(request, training, validation, directory / "fit")
                fit = public.execute_fit(directory / "fit")
                packet = (
                    pipeline.replay_packet(root, directory, request, fit, training)
                    if fit.training.available and fit.parameters is not None
                    else None
                )
                trial.update(
                    fit=fit.model_dump(mode="json"),
                    packet=packet,
                    fit_result_sha256=public.content_sha256(
                        public._read(directory / "fit/result.json")
                    ),
                )
            sealed_write(directory / "result.json", _result(old, entry, trial))
        verify_imports(root, plan)
        reporting.report(root)
        rows = []
        for task in plan["tasks"]:
            if plan[KEY]["entries"][task["task_id"]]["eligible"]:
                result = io.read_round(root, task, 0)
                fit = result["trial"]["fit"]
                rows.append(
                    {
                        "task": task["task_id"],
                        "status": result["status"],
                        "training": fit["training"],
                        "validation": fit["validation"],
                        "residual_packet_available": result["trial"]["packet"]
                        is not None,
                    }
                )
        return sealed_write(
            root / "RECOVERY.json",
            {
                "protocol": POLICY,
                "plan_sha256": plan["artifact_sha256"],
                "optimizer_calls": 0,
                "llm_calls": 0,
                "rows": rows,
                "test_data_opened": False,
                "automatic_next_round": False,
            },
        )
