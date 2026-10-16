"""Offline initializer-information handoffs for sealed saved topology episodes.

No proposer calls or inferred initializer choices occur here. The separate
family test is conditional on the explicitly selected causal runtime contract.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_training as training
from autoformalism.research import saved_topology_repair as saved
from autoformalism.research.actionable_repair_status import build_state
from autoformalism.search import causal_initialization
from autoformalism.search import construction_actionable_repair as repair
from autoformalism.search import topology_initialization as initial
from autoformalism.staged_topology import content_hash

PROTOCOL = "saved-initialization-information-audit-1"
REPO = Path(__file__).resolve().parents[3]


def information_contract(evidence, mode: str):
    """Select a declared runtime hypothesis, never a model-inferred data property."""
    if mode not in {"unknown", "causal_runtime"}:
        raise ValueError("unknown initial-information policy")
    context = evidence.context
    symbols = tuple(
        sorted(
            set(
                context.targets
                + context.auxiliaries
                + context.external_inputs
                + context.fixed_covariates
                + (context.time_symbol,)
            )
        )
    )
    known = mode == "causal_runtime"
    return initial.InitialInformationContract(
        provenance=(
            f"Conditional reuse of {causal_initialization.PROTOCOL}: first public "
            "measurements, fixed covariates and initial time; shared map parameters "
            "or shared initial values. No trajectory ID, future or unlisted history."
            if known
            else "Initial information scope has not been established."
        ),
        scope="recorded_t0" if known else "unknown",
        shared_deterministic=True if known else None,
        complete=True if known else None,
        permitted_symbols=symbols,
    )


def _name(value: str) -> str:
    """Reject path traversal before reading untrusted artifact task names."""
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("unsafe task name")
    return value


def _source(root: Path):
    """Read sealed historical data without requiring it to use today's source code."""
    plan = sealed_read(root / "plan.json")
    if plan.get("protocol") not in {saved.PROTOCOL, saved.PROTOCOL_V2}:
        raise ValueError("requires a saved-topology repair plan")
    if plan.get("test_data_opened") is not False:
        raise ValueError("requires a public-only plan")
    names = [_name(t["task_id"]) for t in plan["tasks"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate task IDs")
    return plan


def _handoff(folder, task, draft, evidence):
    """Accept explicit choices only when bound to these exact inputs."""
    if folder is None:
        return initial.InitializerDependencyHandoff()
    raw = sealed_read(folder / f"{_name(task)}.json")
    if raw.get("topology_sha256") != content_hash(draft.model_dump(mode="json")):
        raise ValueError("initializer handoff topology differs")
    if raw.get("evidence_sha256") != evidence.sha256:
        raise ValueError("initializer handoff training evidence differs")
    return initial.InitializerDependencyHandoff.model_validate(raw["handoff"])


def _preserve_witnesses(report, draft, evidence, previous_questions):
    """Merge causal-prefix rechecks before computing repair completion status."""
    recheck = initial.recheck_closed_witnesses(draft, evidence, previous_questions)
    report["saved_witness_recheck"] = recheck.model_dump(mode="json")
    diagnostics = report["full_report"]["diagnostics"]
    questions = report["advisory_questions"]
    for row in recheck.diagnostics:
        key = (row.target, row.property)
        # Keep a current supported conflict's stable ID. An uncertain current
        # witness must not suppress a supported saved prefix witness.
        if any(
            (q["target"], q["property"]) == key
            and any(
                w.get("error_intervals_disjoint") is True
                for w in q.get("witnesses", ())
            )
            for q in questions
        ):
            continue
        questions[:] = [q for q in questions if (q["target"], q["property"]) != key]
        diagnostics[:] = [d for d in diagnostics if (d["target"], d["property"]) != key]
        diagnostics.append(row.model_dump(mode="json"))
        conditioning = training._conditioning_witness(draft, evidence, row)
        conditioning["comparison"] = (
            "Initial information and all defining forcing knots/covariates were "
            "compared only through each saved witness time. Forcing differences "
            "after that time cannot explain the earlier observation."
        )
        questions.append(
            {
                "target": row.target,
                "property": row.property,
                "scope": list(row.scope),
                "verified": list(row.verified),
                "assumptions": list(row.assumptions),
                "witnesses": [w.model_dump(mode="json") for w in row.witnesses],
                "conditioning": conditioning,
                "question": row.question,
                "question_id": content_hash(
                    {
                        "policy": recheck.protocol,
                        "evidence_sha256": evidence.sha256,
                        "topology_sha256": report["topology_sha256"],
                        "diagnostic": row.model_dump(mode="json"),
                    }
                )[:20],
            }
        )
    uncertain = {"unavailable", "invalid_saved_witness", "uncertainty_limited"}
    for check in recheck.checks:
        if check["status"] not in uncertain or not check.get("target"):
            continue
        key = (check["target"], "closed_subsystem")
        if any((q["target"], q["property"]) == key for q in questions):
            continue
        for row in diagnostics:
            if (row["target"], row["property"]) == key:
                row["status"] = "deferred"
                row["unavailable"] = sorted(
                    set(row.get("unavailable", ()))
                    | {f"saved prefix witness {check['status']}"}
                )
    report["counts"] = dict(Counter(d["status"] for d in diagnostics))


def qualified_feedback(
    draft, evidence, contract, handoff, *, family=False, previous_questions=()
):
    """Keep immutable data identity separate from the initializer hypothesis hash.

    The full diagnostic retains its actual derived evidence hash. The tracking
    envelope retains the original data identity so repairs can be compared without
    pretending that changed initializer declarations changed the training samples.
    """
    family_available = (
        contract.scope == "recorded_t0"
        and contract.shared_deterministic is True
        and contract.complete is True
    )
    derived = initial.derived_evidence(
        draft, evidence, contract, handoff, family=family and family_available
    )
    report = training.compatibility_feedback(
        draft,
        derived,
        identity_policy=training.SEMANTIC_QUESTION_IDENTITY,
    )
    _preserve_witnesses(report, draft, derived, previous_questions)
    report["initialization_evidence_sha256"] = report["evidence_sha256"]
    report["evidence_sha256"] = evidence.sha256
    report["initialization_qualification"] = {
        "kind": "all_permitted_initial_information"
        if family and family_available
        else "declared_only",
        "family_requested": family,
        "family_available": family_available,
        "contract": contract.model_dump(mode="json"),
        "handed_off_rules_are_executable": False,
        "inferred_scientific_choices": False,
    }
    for question in report["advisory_questions"]:
        question["interpretation"] = (
            "Repair a supported conflict or identify an inapplicable hypothesis "
            "with evidence. An explanation alone does not clear the conflict. "
            "This necessary-condition test does not measure fitted accuracy."
        )
        question["initialization_qualification"] = report[
            "initialization_qualification"
        ]
    return report


def _saved_conflicts(start, result):
    """Use the identity-validated final receipt, never unrelated event files.

    Keep the starting witnesses as well as final tracked witnesses. An empty
    tracked list cannot erase the original evidence. Event receipts can belong to
    another episode even when the underlying training evidence is identical.
    """
    original = start["diagnostics"]
    final = result["scientific_state"]
    if final["evidence_sha256"] != original["evidence_sha256"]:
        raise ValueError("saved scientific-state training evidence differs")
    questions = [
        *original.get("advisory_questions", ()),
        *final.get("tracked_conflicts", final.get("advisory_questions", ())),
    ]
    unique = {content_hash(question): question for question in questions}
    return {**final, "tracked_conflicts": list(unique.values())}


def audit(
    source: Path,
    root: Path,
    *,
    mode: str = "unknown",
    handoffs: Path | None = None,
) -> dict:
    """Produce immutable, resumable assessments and an unexecuted prompt preview."""
    if source.resolve() == root.resolve() or source.resolve() in root.resolve().parents:
        raise ValueError("write audit outputs outside the historical source")
    plan = _source(source)
    identity = {
        "source_plan_sha256": plan["artifact_sha256"],
        "mode": mode,
        "source_hashes": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in (
                "src/autoformalism/search/topology_initialization.py",
                "src/autoformalism/research/initialization_handoff_audit.py",
                "src/autoformalism/search/causal_initialization.py",
            )
        },
    }
    sealed_write(root / "identity.json", identity)
    rows = []
    for task in plan["tasks"]:
        name = _name(task["task_id"])
        start = plan["starts"][task["source_task"]]
        result = sealed_read(source / "results" / name / "result.json")
        if result["identity"] != saved.namespace(plan, task):
            raise ValueError("saved result identity differs")
        if result["source_draft_sha256"] != content_hash(start["draft"]):
            raise ValueError("saved starting draft differs")
        if result.get("test_data_opened") is not False:
            raise ValueError("requires a public-only result")
        problem = saved.Problem(plan["cells"][task["benchmark_id"]], start["task"])
        draft = repair.Draft.model_validate(result["draft"])
        evidence = problem.evidence
        contract = information_contract(evidence, mode)
        handoff = _handoff(handoffs, name, draft, evidence)
        baseline = _saved_conflicts(start, result)
        previous = baseline.get(
            "tracked_conflicts", baseline.get("advisory_questions", ())
        )
        inspection = initial.inspect(draft, evidence, contract, handoff)
        declared = qualified_feedback(
            draft, evidence, contract, handoff, previous_questions=previous
        )
        family = qualified_feedback(
            draft,
            evidence,
            contract,
            handoff,
            family=True,
            previous_questions=previous,
        )
        state = build_state(family, baseline, result["assessment"]["eligible"])
        request = initial.request(draft, evidence, contract, handoff)
        prompt = {
            "purpose": "End-of-topology initializer-dependency handoff; preview only.",
            "public_brief": problem.brief.model_dump(mode="json"),
            "model_view": repair.model_view(draft),
            "training_highlights": problem.highlights,
            "initialization": request,
            "qualified_compatibility": {
                "pending_questions": state["pending_questions"],
                "previous_conflicts": state["baseline_outcomes"],
                "qualification": family["initialization_qualification"],
                "acceptance_gate": False,
            },
            "llm_calls": 0,
            "test_data_opened": False,
        }
        output = sealed_write(
            root / "results" / name / "assessment.json",
            {
                "protocol": PROTOCOL,
                "identity": content_hash(identity),
                "task": name,
                "source_result_sha256": result["artifact_sha256"],
                "inspection": inspection,
                "declared_feedback": declared,
                "family_qualified_state": state,
                "source_structural_eligible": result["assessment"]["eligible"],
                "structural_eligibility_changed": False,
                "scientific_correctness": "unverified",
                "llm_calls": 0,
                "optimizer_calls": 0,
                "test_data_opened": False,
            },
        )
        sealed_write(root / "requests" / f"{name}.json", prompt)
        rows.append(
            {
                "task": name,
                "historical_status": result["scientific_state"]["repair_status"],
                "qualified_status": state["repair_status"],
                "declared_counts": declared["counts"],
                "family_counts": family["counts"],
                "assessment_sha256": output["artifact_sha256"],
            }
        )
    summary = sealed_write(
        root / "summary.json",
        {
            "protocol": PROTOCOL,
            **identity,
            "rows": rows,
            "status_counts": dict(Counter(row["qualified_status"] for row in rows)),
            "qualified_diagnostics": dict(
                sum((Counter(row["family_counts"]) for row in rows), Counter())
            ),
            "automatic_followup": False,
            "proposer_choices_inferred": False,
            "llm_calls": 0,
            "optimizer_calls": 0,
            "test_data_opened": False,
        },
    )
    render(root, summary)
    return summary


def render(root: Path, summary: dict) -> None:
    """Render equations, the handoff request and exact conditional feedback."""
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        "<title>Initialization information: prompt review</title>",
        "<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;"
        "padding:0 22px}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f0f4f7;padding:16px}details{border:1px solid #ccd6df;"
        "padding:16px;margin:16px 0}summary{cursor:pointer;font-weight:600}</style>",
        "<h1>Initializer information at the topology boundary</h1>",
        "<p>Preview only: no new calls or model edits. Full initializer formulas "
        "remain in interaction/initialization construction. The new record identifies "
        "initial information, while the separate family test is conditional on "
        "the selected runtime contract. Unknown information is never a pass.</p>",
        f"<p>Selected policy: {html.escape(summary['mode'])}. "
        "All reference-equation audits are excluded from these requests.</p>",
    ]
    for row in summary["rows"]:
        name = row["task"]
        request = sealed_read(root / "requests" / f"{name}.json")
        parts.append(
            "<details><summary>"
            + html.escape(
                f"{name}: {row['historical_status']} → {row['qualified_status']}"
            )
            + "</summary>"
        )
        view = request["model_view"]
        equations = [balance["topology"] for balance in view["balances"]]
        equations.extend(process["topology"] for process in view["processes"])
        parts.append(
            "<h2>Current model</h2><pre>" + html.escape("\n".join(equations)) + "</pre>"
        )
        for title, value in (
            ("New initialization handoff", request["initialization"]),
            (
                "Feedback with witnesses and assumptions",
                request["qualified_compatibility"],
            ),
            ("Training observations", request["training_highlights"]),
        ):
            parts.append(
                f"<h2>{html.escape(title)}</h2><pre>"
                + html.escape(json.dumps(value, indent=2))
                + "</pre>"
            )
        parts.append(
            f'<p><a href="requests/{html.escape(name)}.json">Complete request</a>'
            "</p></details>"
        )
    parts.append("</html>")
    (root / "PROMPTS.html").write_text("\n".join(parts))
