"""Generate and sift five alternatives from each of three T2-easy starting points."""

from __future__ import annotations

import argparse
import html
import json
import tarfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from pydantic import Field, ValidationError

from autoformalism.expressions import ModelValidationError, ValidationContext
from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.construction import MEASURED_RESERVATION, ConstructionClient
from autoformalism.llm.response_revision import PromptPreflightError
from autoformalism.llm.staged_topology import (
    StagedModelSettings,
    atomic_json,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import t2_breadth as science
from autoformalism.research.construction_revision_edits import (
    RevisionValidationError,
    model_view,
)
from autoformalism.research.construction_structural_feedback import consistency_view
from autoformalism.research.construction_structural_revision import (
    apply_structural_revision,
)
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.staged_topology import content_hash
from scripts import propose_meal_insulin_revision as insulin
from scripts import structural_revision_campaign as structural
from scripts import t2_breadth_history as history
from scripts.construction_functions import file_hash

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "t2-three-origin-breadth-2"
INTAKE = "db0b5e78406147a8482c1b05f9d2463dacddbb2937f4f0f542ba9d74700a064c"
SEEDS = tuple(range(100, 105))
GROUPS = ("c15", "c21", "fresh")
MAX_CORRECTIONS = 3
MAX_CALLS = 2 + len(SEEDS) * 2 + MAX_CORRECTIONS * len(GROUPS) * len(SEEDS)
SETTINGS = StagedModelSettings(
    model="openai/gpt-oss-120b",
    model_revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    max_output_tokens=12288,
    maximum_requests=1,
    maximum_total_tokens=32768,
    attempts_per_step=1,
    timeout_seconds=480,
)
ENTRYPOINTS = (
    "scripts/t2_breadth.py",
    "scripts/t2_breadth_history.py",
    "scripts/t2_breadth_scheduler.py",
    "scripts/submit_t2_breadth.py",
    "scripts/hpc/start_t2_breadth_aces.sh",
    "scripts/hpc/run_meal_insulin_revision_aces.sh",
    "scripts/submit_meal_insulin_revision_aces.py",
    "scripts/submit_structural_revision_aces.py",
    "scripts/structural_revision_campaign.py",
    "scripts/construction_feedback_revision.py",
    "scripts/propose_meal_insulin_revision.py",
    "scripts/describe_insulin_boundary.py",
    "scripts/harmony_vocab.py",
    "scripts/construction_functions.py",
)
EVIDENCE_IDS = ["public_task", "fasting_balance", "postmeal_windows"]
FIVE = (
    "Return exactly five independently applicable alternatives. Give each a "
    "different scientific "
    "reason to change the balance. Each starts from the same displayed fitted "
    "parent and must "
    "include its own complete declarations. A sibling's symbols are unavailable."
)
GATES = {
    "weighted_symbolic_initials": True,
    "combined_symbolic_initials": True,
    "common_zero_forcing": True,
    "admission_guard": True,
    "scaled_readout_initials": True,
}


class FiveInsulinRepairs(StrictSchema):
    alternatives: tuple[insulin.BoundaryRevision, ...] = Field(
        min_length=5, max_length=5
    )


def _plan(inputs: dict) -> dict:
    if inputs["artifact_sha256"] != INTAKE:
        raise ValueError("Expected reviewed T2 training intake and both fitted parents")
    return {
        "protocol": PROTOCOL,
        "intake_sha256": INTAKE,
        "groups": list(GROUPS),
        "alternatives_per_group": 5,
        "fresh_topology_seeds": list(SEEDS),
        "maximum_corrections_per_branch": MAX_CORRECTIONS,
        "maximum_generation_requests": MAX_CALLS,
        "settings": SETTINGS.model_dump(mode="json"),
        "runtime_source": runtime_source_hash(),
        "entrypoints": {k: file_hash(REPO / k) for k in ENTRYPOINTS},
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


def prepare(root: Path, source: Path, previous: Path | None = None) -> dict:
    inputs = sealed_read(source)
    plan = _plan(inputs)
    with _lock(root / "prepare-lock"):
        history.bind_source(root, previous, INTAKE)
        sealed_write(
            root / "inputs.json",
            {k: v for k, v in inputs.items() if k != "artifact_sha256"},
        )
        return sealed_write(root / "plan.json", plan)


def load(root: Path):
    saved, inputs = sealed_read(root / "plan.json"), sealed_read(root / "inputs.json")
    expected = _plan(inputs)
    if saved != {**expected, "artifact_sha256": content_hash(expected)}:
        raise ValueError(
            "Frozen intake, settings, source or allocation contract differs."
        )
    return saved, inputs


def _packet(data: dict, group: str) -> tuple[str, dict]:
    brief = data["brief"]
    fasting = next(
        r for r in data["training"]["rows"] if r["trajectory_id"] == "train_000"
    )
    public = {
        "task": brief["scientific_context"],
        "prefix": data["fitting_contract"]["prefix_policy"],
        "fasting_train_000": {
            "meal_rate_range": [
                min(fasting["external_inputs"][science.boundary.MEAL]),
                max(fasting["external_inputs"][science.boundary.MEAL]),
            ],
            **{
                k: [min(fasting["targets"][k]), max(fasting["targets"][k])]
                for k in ("Gp", "U")
            },
        },
        "zero_ingestion_after_90g_train_003": data["glucose_windows"],
        "insulin_training_windows": [
            {
                k: r[k]
                for k in (
                    "trajectory",
                    "meal",
                    "observed_peak_min",
                    "observed_rise_pmol_L",
                )
            }
            for r in data["c15"]["boundary_feedback"]["ingestion_windows"]
        ],
    }
    if group == "fresh":
        return (
            "Construct a new reduced model from the public task and training "
            "observations. "
            "First declare all generated quantities and the sources of each "
            "derivative or algebraic "
            "equation. Each output uses its public name. Observations and forcing "
            "can appear as "
            "sources. Give private states and shared processes distinct names. Use "
            "a generated "
            "process for a shared law. Choose the state dimension, connections, and "
            "scientific "
            "mechanisms. If you model a supplied auxiliary, use a private generated "
            "name, set represents_auxiliary to its public name, and use the "
            "generated name throughout the dynamics. Otherwise set "
            "represents_auxiliary to null. Parameter names and formulas come next.",
            public,
        )
    parent = PrefixFitProblem.model_validate(data[group]["problem"])
    public.update(
        current_model=model_view(parent, scientific_aliases=data["aliases"]),
        scientific_aliases=data["aliases"],
        fitted_dynamics=dict(parent.parameter_guesses),
    )
    if group == "c15":
        public.update(
            depletion=data[group]["boundary_witness"],
            fitted_training_windows=data[group]["boundary_feedback"][
                "ingestion_windows"
            ],
            prior_failure=data["last_failure"]["error"],
            prior_clock_counterexample=data[group]["repair_history"][
                "previous_attempt"
            ]["fixed_parameter_replay"],
        )
        return (
            insulin.BOUNDARY_SYSTEM
            + "\nThe previous draft referenced direct_ingestion (D). "
            "It is absent from this parent. Every proposal that uses it must define "
            "it.",
            public,
        )
    public.update(training_nmse=data[group]["training_nmse"], evidence_ids=EVIDENCE_IDS)
    return (
        "Repair the glucose balance. Expanding the parent's equations gives dGp/dt "
        "+ U = 0 "
        "whenever ingestion is zero. Explain how the proposed balance sustains "
        "fasting glucose "
        "and permits the measured post-meal rise with positive disposal. Preserve "
        "the meaning "
        "of U. You can replace any complete RHS, add or remove private "
        "states/processes, and edit "
        "eligible latent initializers. Provide one complete parent-relative patch "
        "per alternative. "
        "Use only the evidence_ids provided. Changing a private name alone leaves "
        "the mechanism "
        "unchanged. A shared rate should have a single definition.",
        public,
    )


def _problem_outcome(data, group, patch, topology=None):
    """Apply a reply to its parent and run the common admission tests."""
    brief = PublicScientificBrief.model_validate(data["brief"])
    if group == "fresh":
        context = ValidationContext.model_validate(data["evidence"]["context"])
        science.check_topology(topology, context, brief)
        problem = science.assemble(
            topology, science.Laws.model_validate(patch), context
        )
        aliases, inherited, audit = topology.aliases(), None, {}
    else:
        aliases = data["aliases"]
        inherited = PrefixFitProblem.model_validate(data[group]["problem"])
        if group == "c15":
            outcome = insulin.assess(
                {
                    **data[group],
                    "brief": data["brief"],
                    "aliases": aliases,
                    "evidence": data["evidence"],
                },
                patch,
            )
            problem = PrefixFitProblem.model_validate(outcome["problem"])
            audit = {
                k: outcome[k]
                for k in ("audit", "training_consistency", "insulin_boundary_check")
            }
        else:
            edit = science.InScopeRevisionReply.model_validate(patch)
            applied = apply_structural_revision(
                inherited,
                edit,
                evidence_ids=EVIDENCE_IDS,
                brief=brief,
                scientific_aliases=aliases,
                normalize_parent_declarations=True,
            )
            if applied["status"] != "revised":
                raise ValueError("No change to the fitted parent")
            problem, audit = applied["problem"], {"audit": applied["audit"]}
    paths = science.check_paths(problem, brief, aliases)
    compiled, _ = science._validate_problem(problem)
    prefix = data["fitting_contract"]["prefix_policy"]
    permitted = (
        set(prefix.get("future_inputs", ()))
        | set(prefix.get("future_auxiliaries", ()))
        | set(problem.context.fixed_covariates)
    )
    forcing = set(compiled.validated.forcing_symbols)
    if forbidden := forcing - permitted:
        raise ValueError(
            f"Future forcing outside the declared prefix policy: {sorted(forbidden)}"
        )
    audit["future_forcing"] = sorted(forcing)
    balance = science.glucose_balance(problem)
    if balance["status"] == "identically_zero":
        raise RevisionValidationError(
            "ZERO_MEAL_SOURCE_RETAINED",
            "Your assembled glucose equation still enforces dGp/dt = -U during zero "
            "ingestion. "
            "With positive disposal this cannot sustain the fasting level or a "
            "later glucose rise. "
            "Revise a net source or exchange. The observed disposal is sampled.",
            balance=balance,
        )
    if "training_consistency" not in audit:
        audit["training_consistency"] = structural._assessment(
            {"_consistency_evidence": data["evidence"]}, problem, **GATES
        )
        if audit["training_consistency"]["status"] == "conflict":
            raise RevisionValidationError(
                "QUALIFIED_MODEL_DATA_CONFLICT",
                "Public training witness contradicts the model",
                assessment=audit["training_consistency"],
            )
    # Use the same name anchors in the offline comparison for all 15 branches.
    # The fresh prompt never receives this comparison model.
    return {
        "status": "eligible",
        "problem": problem.model_dump(mode="json"),
        "checks": audit,
        "public_paths": paths,
        "glucose_balance": balance,
        "balances": science.balance_view(problem),
        "family": science.alternative_identity(data["c15"]["problem"], problem),
        "modeled_auxiliaries": {
            k: v for k, v in aliases.items() if k in problem.context.auxiliaries
        },
    }


def _feedback(exc):
    if isinstance(exc, ValidationError):
        fields = [
            {"field": list(e["loc"]), "message": e["msg"]}
            for e in exc.errors(include_input=False)[:16]
        ]
        return {
            "code": "RESPONSE_SCHEMA",
            "message": "Correct the listed fields.",
            "details": {"fields": fields},
        }
    return {
        "code": getattr(exc, "code", type(exc).__name__),
        "message": str(exc)[:3500],
        "details": getattr(exc, "details", {}),
    }


def correction_feedback(diagnostic: dict) -> dict:
    """Send a compact training witness. Preserve the full result in the receipt."""
    details = diagnostic.get("details", {})
    if "assessment" not in details:
        return diagnostic
    return {
        **diagnostic,
        "details": {**details, "assessment": consistency_view(details["assessment"])},
    }


def _assess(data, group, raw, stage, topo=None):
    try:
        if stage in {"topology", "topology_correction"}:
            topo = science.Topology.model_validate(raw)
            science.check_topology(
                topo,
                ValidationContext.model_validate(data["evidence"]["context"]),
                PublicScientificBrief.model_validate(data["brief"]),
            )
            return {
                "status": "topology_ready",
                "topology": topo.model_dump(mode="json"),
            }
        if stage == "complete_correction":
            revision = science.FreshCorrection.model_validate(raw)
            topo, raw = (
                revision.topology.model_dump(mode="json"),
                revision.laws.model_dump(mode="json"),
            )
        return _problem_outcome(
            data,
            group,
            raw,
            science.Topology.model_validate(topo) if topo is not None else None,
        )
    except (ValueError, ModelValidationError) as exc:
        return {"status": "rejected", "diagnostic": _feedback(exc)}


def _ask(
    root,
    plan,
    slot,
    seed,
    system,
    packet,
    schema,
    endpoint,
    transport,
    token_transport,
    prefix,
    ceiling,
):
    """One durable generation slot, including failures and interrupted requests."""
    identity = content_hash([plan["artifact_sha256"], slot])
    directory = root / "steps" / slot
    request = {
        "system": system,
        "user": json.dumps(
            packet, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ),
        "response_model": schema,
        "step": slot,
        "attempt": 0,
    }
    key = content_hash(
        [identity, seed, system, request["user"], schema.model_json_schema()]
    )
    path = directory / "reply.json"
    if path.exists():
        reply = sealed_read(path)
        if reply["key"] != key:
            raise ValueError("saved reply belongs to a different prompt")
        if "request_hash" in reply:
            base = (
                (root / "legacy/steps" / slot)
                if reply.get("source_plan")
                else directory
            )
            physical = base / "calls" / (reply["request_hash"] + ".json")
            if file_hash(physical) != reply["physical_receipt_sha256"]:
                raise ValueError(f"Changed physical call record for {slot}")
        return reply
    old = history.prior_reply(root, prefix, slot, seed, request, SETTINGS)
    if old is not None:
        return sealed_write(path, {"key": key, **old})
    # A cache reservation, including an interrupted call, already spent its slot.
    local = list((directory / "calls").glob("*.json"))
    if len(local) > 1:
        raise ValueError(f"Multiple reservations in generation slot {slot}")
    if not local:
        charged = prefix["physical_requests"] + sum(
            1 for _ in (root / "steps").glob("*/calls/*.json")
        )
        if charged >= ceiling:
            raise ValueError(
                f"Physical request ceiling {ceiling} reached before {slot}"
            )
    client = ConstructionClient(
        settings=SETTINGS,
        base_url=endpoint,
        directory=directory / "calls",
        namespace=identity,
        seed=seed,
        reservation_policy=MEASURED_RESERVATION,
        token_transport=token_transport,
        **({"transport": transport} if transport else {}),
    )
    try:
        record = client.call(**request)
    except PromptPreflightError as exc:
        return sealed_write(
            path, {"key": key, "status": "preflight_blocked", "error": str(exc)}
        )
    outcome = {
        **history.visible(record),
        "request_hash": record["request_hash"],
        "physical_receipt_sha256": file_hash(
            directory / "calls" / (record["request_hash"] + ".json")
        ),
    }
    return sealed_write(path, dict(key=key, **outcome))


def _batch_slots(reply):
    if reply["status"] != "visible":
        return [None] * 5
    body = reply["value"]
    if (
        not isinstance(body, dict)
        or set(body) != {"alternatives"}
        or not isinstance(body["alternatives"], list)
        or len(body["alternatives"]) != 5
    ):
        return [None] * 5
    return body["alternatives"]


def _initial_branches(root, plan, data, ask):
    branches = []
    for group in GROUPS:
        system, packet = _packet(data, group)
        if group != "fresh":
            schema = (
                FiveInsulinRepairs if group == "c15" else science.FiveGlucoseRepairs
            )
            reply, ordinal = ask(
                group + "-five", 0, system + "\n" + FIVE, packet, schema
            )
            options = _batch_slots(reply)
        for i, seed in enumerate(SEEDS):
            if group == "fresh":
                name, stage = f"fresh-{seed}", "topology"
                reply, ordinal = ask(
                    name + "-topology", seed, system, packet, science.Topology
                )
                raw = reply.get("value")
            else:
                name, stage, raw = f"{group}-{i}", "patch", options[i]
            outcome = (
                _assess(data, group, raw, stage)
                if raw is not None
                else {
                    "status": "unavailable",
                    "reason": reply.get(
                        "error",
                        reply["status"]
                        if reply["status"] != "visible"
                        else "invalid_five_alternative_envelope",
                    ),
                }
            )
            branch = {
                "name": name,
                "group": group,
                "seed": seed,
                "stage": stage,
                "raw": raw,
                "initial_sampling_seed": seed if group == "fresh" else 0,
                "outcome": outcome,
                "corrections": 0,
                "duplicate_of": None,
                "history": [_event(stage, 0, ordinal, outcome, raw)],
            }
            sealed_write(
                root / "first" / (name + ".json"),
                {
                    "plan": plan["artifact_sha256"],
                    **{
                        k: branch[k]
                        for k in (
                            "name",
                            "group",
                            "seed",
                            "stage",
                            "raw",
                            "initial_sampling_seed",
                        )
                    },
                    "initial": outcome,
                },
            )
            branches.append(branch)
    return branches


def _event(stage, depth, ordinal, outcome, raw):
    return {
        "stage": stage,
        "depth": depth,
        "through_call": ordinal,
        "outcome": outcome["status"],
        "diagnostic": outcome.get("diagnostic"),
        "reply_sha256": content_hash(raw) if raw is not None else None,
    }


def _advance(data, branch, depth, attempted, ask):
    """Spend one correction and supply laws when a new graph passes."""
    name, group, seed = (branch[k] for k in ("name", "group", "seed"))
    system, packet = _packet(data, group)
    while not branch["duplicate_of"]:
        old, previous = branch["outcome"], branch["raw"]
        if old["status"] == "topology_ready":
            branch["topology"] = old["topology"]
            stage, schema = "laws", science.Laws
            instruction, question = (
                LAW_INSTRUCTION,
                {**packet, "topology": branch["topology"]},
            )
        elif old["status"] == "rejected" and branch["corrections"] < depth:
            if depth == 1 and group != "fresh":
                signature = (group, content_hash(previous))
                if signature in attempted:
                    branch["duplicate_of"] = attempted[signature]
                    break
                attempted[signature] = name
            branch["corrections"] += 1
            question = {
                **packet,
                "previous_reply": previous,
                "rejection": correction_feedback(old["diagnostic"]),
                "instruction": "This branch failed admission. Correct the full patch.",
            }
            instruction = system
            if group != "fresh":
                stage = "patch_correction"
                schema = (
                    insulin.BoundaryRevision
                    if group == "c15"
                    else science.InScopeRevisionReply
                )
            elif branch["stage"] in {"topology", "topology_correction"}:
                stage, schema = "topology_correction", science.Topology
            else:
                stage, schema = "complete_correction", science.FreshCorrection
                question["previous_topology"] = branch["topology"]
                instruction = (
                    LAW_INSTRUCTION + " You may revise the topology and laws together."
                )
            if depth > 1:
                question["correction_number"] = depth
                question["earlier_failures"] = [
                    {
                        "stage": h["stage"],
                        "depth": h["depth"],
                        "reason": h["diagnostic"]["message"][:700],
                        "same_reply_as_latest": h["reply_sha256"]
                        == branch["history"][-1]["reply_sha256"],
                    }
                    for h in branch["history"][:-1]
                    if h["diagnostic"]
                ]
                question["instruction"] += (
                    " Keep useful edits. Address the latest rejection and its witness."
                )
        else:
            break
        slot = name + "-" + stage
        if depth > 1 and stage.endswith("correction"):
            slot += f"-{depth}"
        reply, ordinal = ask(slot, seed, instruction, question, schema)
        raw = reply.get("value")
        outcome = (
            _assess(
                data,
                group,
                raw,
                stage,
                branch.get("topology") if stage == "laws" else None,
            )
            if reply["status"] == "visible"
            else {
                "status": "unavailable",
                "reason": reply.get("error", reply["status"]),
            }
        )
        if (
            stage == "complete_correction"
            and isinstance(raw, dict)
            and isinstance(raw.get("topology"), dict)
        ):
            # The next whole-model repair refers to the topology actually proposed.
            branch["topology"] = raw["topology"]
        branch.update(stage=stage, raw=raw, outcome=outcome)
        branch["history"].append(
            _event(stage, branch["corrections"], ordinal, outcome, raw)
        )


def _result(branch, plan):
    return {
        "plan": plan["artifact_sha256"],
        "name": branch["name"],
        "group": branch["group"],
        "initial_sampling_seed": branch["initial_sampling_seed"],
        "correction_seed": branch["seed"],
        "correction_duplicate_of": branch["duplicate_of"],
        "correction_count": branch["corrections"],
        "correction_used": bool(branch["corrections"]),
        "history": branch["history"],
        **branch["outcome"],
    }


def run(root: Path, endpoint: str, *, transport=None, token_transport=None):
    """Visit all branches at one depth before starting the next."""
    from urllib.parse import urlsplit

    u = urlsplit(endpoint)
    if (
        u.scheme != "http"
        or u.hostname != "127.0.0.1"
        or u.path not in ("", "/")
        or u.username
        or u.password
        or u.query
        or u.fragment
    ):
        raise ValueError("An unauthenticated loopback vLLM endpoint is required")
    with _lock(root / "worker-lock"):
        plan, data = load(root)
        prefix = history.snapshot(root)
        ledger = []
        ceiling = 27

        def ask(slot, seed, system, packet, schema):
            reply = _ask(
                root,
                plan,
                slot,
                seed,
                system,
                packet,
                schema,
                endpoint,
                transport,
                token_transport,
                prefix,
                ceiling,
            )
            if "request_hash" in reply:
                ledger.append(
                    {
                        "call": len(ledger) + 1,
                        "slot": slot,
                        "request_hash": reply["request_hash"],
                        "source_plan": reply.get(
                            "source_plan", plan["artifact_sha256"]
                        ),
                    }
                )
            return reply, len(ledger)

        branches = _initial_branches(root, plan, data, ask)
        attempted = {}
        for depth in range(1, MAX_CORRECTIONS + 1):
            ceiling = 27 + (depth - 1) * len(GROUPS) * len(SEEDS)
            dest = root / "depths" / str(depth)
            for branch in branches:
                _advance(data, branch, depth, attempted, ask)
                row = _result(branch, plan)
                sealed_write(dest / (branch["name"] + ".json"), row)
            if depth == 1:
                used = {
                    call["slot"]
                    for call in ledger
                    if call["source_plan"] == history.V1_PLAN
                }
                charged = {
                    slot
                    for slot, files in prefix["slots"].items()
                    if any(f.startswith("calls/") for f in files)
                }
                if used != charged or len(ledger) > 27:
                    raise ValueError(
                        "Original calls differ from the one-correction checkpoint"
                    )
            rows = [_result(b, plan) for b in branches]
            order = sealed_write(dest / "calls.json", {"calls": list(ledger)})
            sealed_write(
                dest / "summary.json",
                {
                    "depth": depth,
                    "call_order_sha256": order["artifact_sha256"],
                    "physical_requests": len(ledger),
                    "eligible": sum(r["status"] == "eligible" for r in rows),
                    "groups": {
                        g: dict(Counter(r["status"] for r in rows if r["group"] == g))
                        for g in GROUPS
                    },
                    "distinct_representatives": len(_balance_groups(rows)),
                    "corrections_attempted": sum(r["correction_count"] for r in rows),
                },
            )
        sealed_write(root / "call-order.json", {"calls": ledger})
        for branch in branches:
            sealed_write(
                root / "results" / (branch["name"] + ".json"), _result(branch, plan)
            )
        return report(root)


LAW_INSTRUCTION = (
    "Write one explicit analytic RHS for every proposed graph component, using the "
    "declared scientific sources. Declare shared dynamics coefficients and choose a "
    "causal initializer "
    "for each private differential state. Measured differential targets start from "
    "their recorded values. "
    "Use shared_training or trajectory for independent latent roots, or a map of "
    "causal public readings "
    "and other initial states. Use the declared 8-minute prefix for inference of "
    "trajectory roots. "
    "The runtime supplies parameter guesses. Consider fasting balance, the delayed "
    "meal response, and "
    "the signs of the complete rates at zero concentration. Use the restricted "
    "analytic grammar."
)


def _rows(root: Path, plan: dict) -> list[dict]:
    names = [f"{g}-{i}" for g in GROUPS[:2] for i in range(5)] + [
        f"fresh-{i}" for i in SEEDS
    ]
    rows = []
    for name in names:
        path = root / "results" / (name + ".json")
        if path.exists():
            row = sealed_read(path)
            if row["plan"] != plan["artifact_sha256"] or row["name"] != name:
                raise ValueError(f"Unbound branch receipt: {name}")
            rows.append(row)
    return rows


def _balance_groups(rows):
    """Group equivalent admitted branches using the original sifting policy."""
    groups = []
    for row in rows:
        if row["status"] != "eligible":
            continue
        aliases = content_hash(row["modeled_auxiliaries"])
        keys = {
            (kind, value, aliases)
            for kind, value in (
                ("balances", row["balances"].get("balance_sha256")),
                ("family", row["family"]),
            )
            if value
        }
        members = [row["name"]]
        separate = []
        for group_keys, group_names in groups:
            if keys & group_keys:
                keys.update(group_keys)
                members.extend(group_names)
            else:
                separate.append((group_keys, group_names))
        groups = [*separate, (keys, sorted(members))]
    members_by_rep = {
        names[0]: names for _, names in sorted(groups, key=lambda g: g[1])
    }
    return members_by_rep


def report(root: Path) -> dict:
    """Report terminal states, actual expenditures and each stopping point."""
    plan, inputs = load(root)
    rows = _rows(root, plan)
    members_by_rep = _balance_groups(rows)
    representatives = list(members_by_rep)
    receipts = [
        json.loads(p.read_text())
        for base in (root / "legacy/steps", root / "steps")
        for p in sorted(base.glob("*/calls/*.json"))
    ]
    if len(receipts) > MAX_CALLS:
        raise ValueError("Campaign physical request ceiling exceeded")
    yield_by_origin = {}
    for origin in GROUPS:
        accepted = [
            r for r in rows if r["group"] == origin and r["status"] == "eligible"
        ]
        after = sum(r["correction_used"] for r in accepted)
        yield_by_origin[origin] = {
            "planned": 5,
            "first_pass": len(accepted) - after,
            "after_correction": after,
        }
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status": "complete" if len(rows) == 15 else "incomplete",
        "expected": 15,
        "recorded": len(rows),
        "unavailable_branches": sum(row["status"] == "unavailable" for row in rows),
        "groups": {
            g: dict(Counter(r["status"] for r in rows if r["group"] == g))
            for g in GROUPS
        },
        "admission_by_origin": yield_by_origin,
        "eligible_first_pass": sum(g["first_pass"] for g in yield_by_origin.values()),
        "eligible_after_correction": sum(
            g["after_correction"] for g in yield_by_origin.values()
        ),
        "corrections_attempted": sum(r["correction_count"] for r in rows),
        "inherited_physical_requests": len(
            list((root / "legacy/steps").glob("*/calls/*.json"))
        ),
        "depth_progress": {
            str(d): sum(
                p.stem in {*history.BRANCHES, *history.FRESH}
                for p in (root / "depths" / str(d)).glob("*.json")
            )
            for d in range(1, MAX_CORRECTIONS + 1)
        },
        "depth_comparison": [
            sealed_read(p) for p in sorted((root / "depths").glob("*/summary.json"))
        ],
        "physical_requests": len(receipts),
        "usage_missing": sum(r.get("observed_total_tokens") is None for r in receipts),
        "observed_tokens": sum(r.get("observed_total_tokens") or 0 for r in receipts),
        "provider_outcomes": dict(Counter(r["status"] for r in receipts)),
        "representatives": representatives,
        "balance_groups": list(members_by_rep.values()),
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
    if len(rows) == 15:
        # An early shallow finish leaves room for deeper calls in the first 27.
        cutoff = min(27, len(receipts))
        admitted = [
            r
            for r in rows
            if r["status"] == "eligible" and r["history"][-1]["through_call"] <= cutoff
        ]
        summary["first_27_physical_calls"] = {
            "through_call": cutoff,
            "eligible": len(admitted),
            "representatives": list(_balance_groups(admitted)),
        }
    # Freeze representatives once every branch has a disposition.
    if len(rows) == 15:
        for name in representatives:
            row = next(r for r in rows if r["name"] == name)
            sealed_write(
                root / "handoff" / (name + ".json"),
                {
                    "branch": name,
                    "equivalent_branches": members_by_rep[name],
                    "origin": row["group"],
                    "plan_sha256": plan["artifact_sha256"],
                    "source_receipt": row["artifact_sha256"],
                    "problem": row["problem"],
                    "model_view": model_view(
                        PrefixFitProblem.model_validate(row["problem"])
                    ),
                    "public_paths": row["public_paths"],
                    "checks": row["checks"],
                    "glucose_balance": row["glucose_balance"],
                    "training_file": "inputs.json#training",
                    "prefix_policy": inputs["fitting_contract"]["prefix_policy"],
                    "training_sha256": inputs["fitting_contract"]["training_sha256"],
                    "parent_reference": (
                        inputs["c15"]["result_identity"]
                        if row["group"] == "c15"
                        else {"anchor_sha256": inputs["c21"]["anchor_sha256"]}
                        if row["group"] == "c21"
                        else None
                    ),
                    "qualification": (
                        "Fit on matched training rows. Verify replay NMSE, "
                        "fasting balance, insulin minima, timing and mechanisms. "
                        "Check deferred items independently."
                    ),
                    "fitting_status": "pending",
                    "scientific_correctness_established": False,
                },
            )
        atomic_json(root / "summary.json", summary)
    return summary


def _page(title: str) -> list[str]:
    return [
        '<!doctype html><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1">',
        "<style>body{font:16px/1.5 system-ui;max-width:68rem;margin:3rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "padding:1rem;background:#f5f6f8}summary{cursor:pointer}"
        "table{border-collapse:collapse}td,th{padding:.4rem;border:1px solid #ccc}"
        "</style>",
        f"<title>{title}</title><h1>{title}</h1>",
    ]


def _details(label, value):
    return (
        f"<details><summary>{html.escape(label)}</summary><pre>"
        + html.escape(json.dumps(value, indent=2, ensure_ascii=False))
        + "</pre></details>"
    )


def preview(root: Path):
    """Render exactly what each of the three initial origins receives."""
    _, data = load(root)
    parts = _page("T2-easy breadth prompts")
    for group in GROUPS:
        system, packet = _packet(data, group)
        if group != "fresh":
            system += "\n" + FIVE
        parts.extend(
            [
                f"<h2>{group}</h2><pre>{html.escape(system)}</pre>",
                _details("Complete initial request", packet),
            ]
        )
    parts.extend(
        [
            "<h2>Fresh function construction</h2><pre>",
            html.escape(LAW_INSTRUCTION),
            "</pre><h2>Correction</h2>",
            "<p>Up to three corrections per branch, one round across all branches "
            "at a time. Each receives its previous reply and the "
            "deterministic rejection. A fresh formula correction may also "
            "change the topology.</p>",
            _details("Example of a recorded rejection", data["last_failure"]),
        ]
    )
    path = root / "PROMPTS.html"
    path.write_text("\n".join(parts))
    return {"preview": str(path)}


def _render(root: Path, summary: dict):
    """Expose each branch's dispositions, equations and checkable witnesses."""
    parts = _page("T2-easy construction results")
    parts.append(
        f"<p>{summary['recorded']} of 15 branches recorded. "
        f"{len(summary['representatives'])} representatives for fitting.</p>"
    )
    parts.append(_details("First-pass yield, correction yield and costs", summary))
    for row in _rows(root, sealed_read(root / "plan.json")):
        parts.append(
            f"<h2>{html.escape(row['name'])} · {html.escape(row['status'])}</h2>"
        )
        parts.append(_details("Stage outcomes and repair feedback", row["history"]))
        if row["status"] == "eligible":
            eqs = model_view(PrefixFitProblem.model_validate(row["problem"]))[
                "equations"
            ]
            parts.append(
                "<pre>" + html.escape("\n".join(e["display"] for e in eqs)) + "</pre>"
            )
            parts.append(_details("Expanded balances and readouts", row["balances"]))
            parts.append(_details("Public mechanism paths", row["public_paths"]))
            parts.append(_details("Training and boundary checks", row["checks"]))
        elif row["status"] == "unavailable":
            parts.append(f"<p>{html.escape(row['reason'])}</p>")
    (root / "REPORT.html").write_text("\n".join(parts))


def inspect(root: Path):
    with _lock(root / "inspection-lock"):
        summary = report(root)
        _render(root, summary)
        archive = root / "inspection.tar.gz"
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            encoded = json.dumps(summary, indent=2).encode()
            member = tarfile.TarInfo("summary.json")
            member.size = len(encoded)
            tar.addfile(member, BytesIO(encoded))
            for name in (
                "inputs.json",
                "plan.json",
                "execution.json",
                "PROMPTS.html",
                "REPORT.html",
                "legacy",
                "depths",
                "call-order.json",
                "predecessor.json",
                "first",
                "results",
                "steps",
                "handoff",
                "submissions",
                "runtime",
                "logs",
            ):
                start = root / name
                paths = sorted(start.rglob("*")) if start.is_dir() else [start]
                for path in paths:
                    if (
                        path.is_file()
                        and not path.is_symlink()
                        and not path.name.endswith(".tmp")
                        and not path.name.startswith(".")
                    ):
                        tar.add(
                            path, arcname=str(path.relative_to(root)), recursive=False
                        )
        temporary.replace(archive)
        return {**summary, "archive": str(archive)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "status", "preview", "run", "inspect"))
    p.add_argument("--source", type=Path)
    p.add_argument("--previous", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--endpoint")
    p.add_argument("--runtime-record", type=Path)
    p.add_argument("--model-size", choices=("120b",))
    args = p.parse_args()
    if args.action == "prepare" and not args.source:
        p.error("prepare requires --source")
    if args.action == "run" and (not args.runtime_record or not args.endpoint):
        p.error("run requires --endpoint and --runtime-record from the ACES worker")
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(root, args.source, args.previous)
        preview(root)
    elif args.action == "run":
        from scripts.submit_t2_breadth import bind

        bind(root, args.runtime_record, args.endpoint)
        result = run(root, args.endpoint)
    elif args.action == "inspect":
        result = inspect(root)
    elif args.action == "preview":
        result = preview(root)
    else:
        result = report(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
