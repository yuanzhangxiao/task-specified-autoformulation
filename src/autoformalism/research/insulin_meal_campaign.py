"""Matched fits of candidate 15's insulin equation on public training data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "candidate-15-insulin-meal-control-1"
ARMS = ("parent", "child")
SOURCE_PLAN = "ad70b882c9aed0a3f37b252c52e5ad3ad97af658de01ef7c2e5c618af56fa458"
SOURCE_TERMINAL = "dcc9c9960aa5487d51b60dd89c50cbf1eb2ef68dc4633c4783baef4370723453"
CONTEXT_FILE = "a7448095f17f1572e17b9a821c0af57602d5a5135285c001fbfe5d48eac54b35"
EVIDENCE_FILE = "028e0ba65c052601b1c97c8b18ef65453a19a442052cc4d765c6b978b02fa5e4"
SOURCE_PACKET = "5d579c5fe048c0ba445e85eb1276411d2e0c42ba9b18a2d5db16a70fd25bd1cb"


def _pinned_json(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"unexpected public source file: {path.name}")
    return json.loads(raw)


def source_packet(source: Path, evidence: Path) -> dict:
    """Project only approved public inputs from the reviewed Orion endpoint."""
    plan = read_seal(source / "plan.json")
    if public.content_sha256(plan) != SOURCE_PLAN:
        raise ValueError("expected the reviewed November 5 reliability plan")
    data = read_seal(source / "inputs.json")
    task = plan["tasks"][0]
    terminal = read_seal(source / "results" / task["task_id"] / "result.json")
    if (
        public.content_sha256(data) != plan["inputs_sha256"]
        or public.content_sha256(terminal) != SOURCE_TERMINAL
        or terminal["identity"] != {"plan_sha256": SOURCE_PLAN, "task": task}
        or not terminal["retained"]["original_equations_verified"]
    ):
        raise ValueError("source endpoint or input identity differs")
    context = _pinned_json(evidence / "public-context.json", CONTEXT_FILE)
    witness = _pinned_json(evidence / "compatibility.json", EVIDENCE_FILE)
    if (
        data["training"]["name"] != "train"
        or witness["split"] != "train"
        or public.content_sha256(data["training"]) != context["training_sha256"]
    ):
        raise ValueError("public training identities differ")
    return {
        "protocol": PROTOCOL,
        "source_plan_sha256": SOURCE_PLAN,
        "source_terminal_sha256": SOURCE_TERMINAL,
        "problem": data["records"][task["record_index"]]["problem"],
        "retained": terminal["retained"],
        "numerical": data["numerical"],
        "training": data["training"],
        "compatibility": witness,
        "public_brief": context["public_brief"],
        "public_task": context["public_task"],
        "scientific_aliases": context["scientific_aliases"],
        "prefix_policy": context["fixed_fitting_prefix_policy"],
    }


def _plan(data: dict, policy: dict) -> dict:
    from autoformalism.fitting import insulin_meal_comparison as fit

    if public.content_sha256(data) != SOURCE_PACKET:
        raise ValueError("expected the pinned candidate-15 training packet")
    settings = fit.Policy.model_validate(policy)
    parent = PrefixFitProblem.model_validate(data["problem"])
    child = fit.child_problem(parent)
    return {
        "protocol": PROTOCOL,
        "inputs_sha256": SOURCE_PACKET,
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": settings.model_dump(mode="json"),
        "arms": list(ARMS),
        "problem_hashes": {
            "parent": public.content_sha256(parent),
            "child": public.content_sha256(child),
        },
        "binding_hashes": {
            "parent_problem_sha256": public.content_sha256(parent),
            **{
                f"{name}_sha256": public.content_sha256(data[name])
                for name in ("training", "numerical", "retained")
            },
        },
        "objective": "original_full_training_reconstruction",
        "automatic_followup": False,
        "test_data_opened": False,
    }


def prepare(root: Path, packet: Path, policy: dict) -> dict:
    """Freeze a matched budget and arm identities without fitting on a login node."""
    data = read_seal(packet)
    plan = _plan(data, policy)
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"plan_sha256": public.content_sha256(plan), "arms": list(ARMS)}


def verify(root: Path, *, runtime: bool = False) -> tuple[dict, dict]:
    """Verify the exact intake, policy and code before fitting or routing a result."""
    data, plan = read_seal(root / "inputs.json"), read_seal(root / "plan.json")
    expected = _plan(data, plan["policy"])
    if not runtime:
        expected["runtime"] = plan["runtime"]
    if expected != plan:
        raise ValueError(
            "insulin meal comparison plan or execution environment differs"
        )
    return plan, data


def _identity(plan: dict, arm: str) -> dict:
    return {
        "plan_sha256": public.content_sha256(plan),
        "arm": arm,
        "problem_sha256": plan["problem_hashes"][arm],
    }


def run(root: Path, index: int) -> dict:
    from autoformalism.fitting import insulin_meal_comparison as fit
    from autoformalism.fitting.prefix_initialization import NumericalPolicy

    if index not in (0, 1):
        raise ValueError("expected array index 0 or 1")
    plan, data = verify(root, runtime=True)
    arm = ARMS[index]
    return fit.run_arm(
        root / "results" / arm,
        _identity(plan, arm),
        PrefixFitProblem.model_validate(data["problem"]),
        PublicSplit.model_validate(data["training"]),
        NumericalPolicy.model_validate(data["numerical"]),
        data["retained"],
        arm,
        fit.Policy.model_validate(plan["policy"]),
    )


def _terminals(root: Path, plan: dict) -> dict:
    from autoformalism.fitting import insulin_meal_comparison as fit

    policy = fit.Policy.model_validate(plan["policy"])
    terminals = {}
    for arm in ARMS:
        folder = root / "results" / arm
        if (folder / "result.json").is_file():
            binding = read_seal(folder / "binding.json")
            if any(binding.get(k) != v for k, v in plan["binding_hashes"].items()):
                raise ValueError(f"{arm} fitting inputs differ from the frozen plan")
            terminals[arm] = fit.read_arm(folder, _identity(plan, arm), arm, policy)
    return terminals


def comparison(root: Path, plan: dict) -> tuple[dict, dict]:
    from autoformalism.research import meal_revision_metrics as metrics

    terminals = _terminals(root, plan)
    summaries = {
        arm: {
            k: terminal.get(k)
            for k in (
                "ready_for_comparison",
                "operation_statuses",
                "search_termination",
                "search_calls_completed",
                "operation_errors",
                "retained_stage",
            )
        }
        for arm, terminal in terminals.items()
    }
    for arm, terminal in terminals.items():
        endpoint = terminal.get("retained_endpoint") or {}
        summaries[arm].update(
            training_nmse=endpoint.get("training_nmse"),
            direct_insulin_meal_gain=(
                1.0 if arm == "parent" else endpoint.get("parameters", {}).get("c_I")
            ),
        )
    ready = len(terminals) == len(ARMS) and all(
        t["ready_for_comparison"] is True for t in terminals.values()
    )
    paired = (
        metrics.compare(terminals["parent"]["metrics"], terminals["child"]["metrics"])
        if ready
        else None
    )
    return {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if len(terminals) == len(ARMS) else "incomplete",
        "arms": summaries,
        "ready_for_comparison": ready,
        "comparison": paired,
        "automatic_followup": False,
        "test_data_opened": False,
    }, terminals


def proposer_inputs(campaign_root: Path) -> dict:
    """Prepare the verified child's evidence when its early-peak criterion holds."""
    from autoformalism.fitting import insulin_meal_comparison as fit

    plan, data = verify(campaign_root)
    result, terminals = comparison(campaign_root, plan)
    pair = result["comparison"]
    if not pair or pair["routing"]["eligible_for_proposer"] is not True:
        raise ValueError(
            "verified matched result does not request an insulin-law revision"
        )
    child = fit.child_problem(PrefixFitProblem.model_validate(data["problem"]))
    endpoint = terminals["child"]["retained_endpoint"]
    payload = child.model_dump(mode="json")
    payload["parameter_guesses"] = {
        name: endpoint["parameters"][name] for name in child.parameter_guesses
    }
    return {
        "fitted_problem": PrefixFitProblem.model_validate(payload),
        "public_brief": data["public_brief"],
        "scientific_aliases": data["scientific_aliases"],
        "consistency_evidence": data["compatibility"],
        "comparison": pair | {"child_metrics": terminals["child"]["metrics"]},
        "fitting_contract": {
            "prefix_policy": data["prefix_policy"],
            "matched_policy": plan["policy"],
            "training_sha256": plan["binding_hashes"]["training_sha256"],
        },
        "result_identity": {
            "plan_sha256": result["plan_sha256"],
            "comparison_sha256": public.content_sha256(pair),
            "endpoint_sha256": public.content_sha256(endpoint),
            "child_problem_sha256": plan["problem_hashes"]["child"],
        },
    }


def report(root: Path) -> dict:
    """Rebuild the comparison from durable terminal receipts without rerunning a fit."""
    from autoformalism.research import meal_revision_metrics as metrics

    plan, _ = verify(root)
    result, terminals = comparison(root, plan)
    lines = [
        "# Candidate 15 insulin meal comparison",
        "",
        "Full 16-trajectory training reconstruction using fitted training initials.",
        "",
    ]
    for arm in ARMS:
        row = result["arms"].get(arm)
        if row:
            lines.append(
                f"- {arm}: {row['search_termination']}, retained training NMSE "
                f"{row['training_nmse']}, direct insulin meal gain "
                f"{row['direct_insulin_meal_gain']}, retained stage "
                f"{row['retained_stage']}"
            )
            if row["operation_errors"]:
                lines.append(f"  Operation errors: {row['operation_errors']}")
        else:
            lines.append(f"- {arm}: pending")
    if result["ready_for_comparison"]:
        lines += [
            "",
            metrics.render_comparison(
                terminals["parent"]["metrics"], terminals["child"]["metrics"]
            ),
        ]
    else:
        lines += [
            "",
            "Paired assessment awaits two qualified, independently replayed fits.",
        ]
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", result)
        (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return result


def inspect(root: Path) -> Path:
    """Collect the approved training receipts, metrics and report for offline review."""
    from autoformalism.research.construction_fit_qualification import _archive

    report(root)
    with public._lock(root / "report"):
        return _archive(root)
