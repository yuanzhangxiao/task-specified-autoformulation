"""Replay one saved sibling response with exact parameter declaration inheritance."""

from __future__ import annotations

from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting import fit_convergence as lineage
from autoformalism.fitting import fit_residual_feedback as evidence
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import prefit_numerical_sibling as sibling
from autoformalism.rebuttal.prefit_fit_handoff import _disjoint
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.search import numerical_sibling as review
from autoformalism.search.parameter_reuse import POLICY

PROTOCOL = "sibling-parameter-replay-1"


def _historical(source: Path) -> tuple[dict, dict, dict]:
    """Verify historical hashes and strict decisions without resuming that run."""
    plan = sealed_read(source / "plan.json")
    config = sibling.SiblingConfig.model_validate(plan["config"])
    if plan["protocol"] != sibling.PROTOCOL:
        raise ValueError("unknown historical sibling protocol")
    sibling._check_reservation(source, plan)
    frozen = lineage._read_freeze(source / "evidence/freeze.json")
    if (
        frozen["identity"] != plan["residual_identity"]
        or frozen["source_sha256"] != plan["source_sha256"]
        or frozen["runtime"] != plan["runtime"]
        or public.content_sha256(frozen["seed"]) != plan["seed_sha256"]
        or frozen["selection"] != config.selection.model_dump(mode="json")
        or evidence._seed(
            Path(plan["paths"]["parent"]),
            Path(plan["paths"]["continuation"]),
            config.selection,
        )
        != frozen["seed"]
        or frozen["seed"]["parent"]["freeze"]["lowered_candidate"]
        != plan["bundle"]["candidate"]
    ):
        raise ValueError("historical seed/plan differs")
    residual = evidence._result(source / "evidence", frozen)
    if residual["status"] != "ready":
        raise ValueError("saved training evidence is not ready")
    # Default strict behavior is deliberately unchanged. This checks cached
    # request bodies, raw replies, event hashes, order and historical rejections.
    sibling._state(source, plan, residual)
    state = sealed_read(source / "results/state.json")
    if state["status"] != "complete" or state["decision"] is not None:
        raise ValueError("replay requires a terminal rejected episode")
    if (source / "child_fit").exists():
        raise ValueError("historical episode already contains a child allocation")
    return plan, residual, state


def _decision(plan: dict, residual: dict, raw: dict) -> dict:
    """Change only declaration interpretation; retain the saved hypothesis/citations."""
    reply, _ = review.checked_reply(
        raw, residual["packet"], plan["config"]["feedback_policy"]
    )
    if reply.action != "revise_function":
        raise ValueError("selected saved response is not a function revision")
    try:
        decision = review.apply_revision(
            plan["bundle"],
            plan["bindings"],
            residual["packet"],
            raw,
            policy=plan["config"]["feedback_policy"],
            inherit_existing_parameters=True,
        )
        return {"accepted": decision["outcome"] == "committed", "decision": decision}
    except (ValueError, KeyError, TypeError, ModelValidationError) as error:
        return {"accepted": False, "decision": None, "error": str(error)}


def _reservation(source: Path, state_hash: str) -> Path:
    return source.parent / ".sibling-parameter-replays" / state_hash / POLICY


def prepare(source: Path, root: Path, *, attempt: int = 0) -> dict:
    """Freeze one preselected saved attempt; make no provider or fitting calls."""
    source, root = source.resolve(), root.resolve()
    _disjoint(root, source)
    old, residual, state = _historical(source)
    _disjoint(
        root,
        Path(old["paths"]["parent"]).parent,
        Path(old["paths"]["continuation"]).parent,
        Path(old["paths"]["source"]),
        Path(old["paths"]["construction"]),
    )
    events = [e for e in state["attempts"] if e["attempt"] == attempt]
    if len(events) != 1 or events[0]["accepted"]:
        raise ValueError("select one recorded rejected attempt")
    event = events[0]
    outcome = _decision(old, residual, event["response"])
    body = {
        "protocol": PROTOCOL,
        "parameter_policy": POLICY,
        "source_root": str(source),
        "source_plan_sha256": old["artifact_sha256"],
        "source_state_sha256": state["artifact_sha256"],
        "packet_sha256": residual["packet"]["packet_sha256"],
        "attempt": attempt,
        "saved_event": event,
        "outcome": outcome,
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
    }
    ledger = _reservation(source, state["artifact_sha256"])
    with public._lock(root), public._lock(ledger):
        sealed_write(
            ledger / "reservation.json",
            {"output": str(root), "replay": public.content_sha256(body)},
        )
        sealed_write(root / "plan.json", body)
        return _report(root, *verify(root))


def verify(root: Path) -> tuple[dict, dict, dict]:
    """Recheck the exact response, immutable history and new runtime policy."""
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["parameter_policy"] != POLICY
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("parameter replay source, runtime or policy changed")
    source = Path(plan["source_root"])
    old, residual, state = _historical(source)
    ledger = sealed_read(
        _reservation(source, state["artifact_sha256"]) / "reservation.json"
    )
    body = {k: v for k, v in plan.items() if k != "artifact_sha256"}
    if ledger["output"] != str(root.resolve()) or ledger[
        "replay"
    ] != public.content_sha256(body):
        raise ValueError("parameter replay reservation differs")
    if (
        old["artifact_sha256"] != plan["source_plan_sha256"]
        or state["artifact_sha256"] != plan["source_state_sha256"]
        or residual["packet"]["packet_sha256"] != plan["packet_sha256"]
        or state["attempts"][plan["attempt"]] != plan["saved_event"]
        or _decision(old, residual, plan["saved_event"]["response"]) != plan["outcome"]
    ):
        raise ValueError("saved response or replay decision changed")
    return plan, old, residual


def _contract(plan: dict, old: dict, residual: dict) -> dict:
    contract = sibling._child_contract(old, residual, plan["outcome"]["decision"])
    contract["lineage"].update(
        parameter_replay_sha256=plan["artifact_sha256"],
        historical_state_sha256=plan["source_state_sha256"],
        saved_attempt=plan["attempt"],
        parameter_policy=POLICY,
    )
    return contract


def _report(root: Path, plan: dict, old: dict, residual: dict) -> dict:
    outcome = plan["outcome"]
    child = None
    if (root / "child_fit/freeze.json").exists():
        if not outcome["accepted"]:
            raise ValueError("rejected replay contains child fit")
        expected = sibling_fit._freeze(**_contract(plan, old, residual))
        if public._read(root / "child_fit/freeze.json") != expected:
            raise ValueError("child belongs to a different replay decision")
        child = sibling_fit.inspect_child_fit(root / "child_fit")
    result = {
        "protocol": PROTOCOL,
        "status": "complete"
        if child and child["result"]
        else ("ready_for_fit" if outcome["accepted"] else "rejected"),
        "parameter_policy": POLICY,
        "plan_sha256": plan["artifact_sha256"],
        "source_state_sha256": plan["source_state_sha256"],
        "saved_attempt": plan["attempt"],
        "historical_accepted": plan["saved_event"]["accepted"],
        **outcome,
        "parent": residual["seed"]["state"],
        "child": child,
        "live_llm_calls": 0,
        "historical_state_modified": False,
        "parameter_fitting_performed": (root / "child_fit/started.json").exists(),
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_branch_selection": False,
        "limitation": "Exact saved-response replay, not a fresh proposer success rate. "
        "An accepted RHS is not scientifically certified. Parent and child have "
        "different fitting budgets; no equal-compute advantage is inferred.",
    }
    public._write(root / "summary.json", result)
    return result


def report(root: Path) -> dict:
    """Report verified saved results without allocating fitting or provider work."""
    with public._lock(root):
        return _report(root, *verify(root))


def fit(root: Path) -> dict:
    """Fit only an accepted replay, once, using the unchanged child fitter."""
    with public._lock(root):
        plan, old, residual = verify(root)
        if plan["outcome"]["accepted"]:
            sibling_fit.prepare_child_fit(
                directory=root / "child_fit", **_contract(plan, old, residual)
            )
            sibling_fit.execute_child_fit(root / "child_fit")
        return _report(root, plan, old, residual)
