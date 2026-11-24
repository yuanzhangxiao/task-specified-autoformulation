"""Compare 20B, trajectory-ranked functions and their combination on fixed graphs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import (
    StagedModelSettings,
    strict_provider_schema,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.staged_topology import content_hash
from scripts import t2_breadth as pilot
from scripts import trajectory_function_prompts as prompts
from scripts import trajectory_function_revision_prompts as revision_prompts
from scripts.construction_functions import file_hash

PROTOCOL = "trajectory-function-comparison-1"
TRAIN = "7e4bb4712fdf778b0174a4945f89f9f30a8d9c9fcc648591096db03c60fa4ecd"
VALIDATION = "bfd00cc19e3194056166119c634576a3278382b50d13636e6687a8321a53801d"
FIXED = "fixed"
SOURCE_REVIEW = "source-review"
BASIN_REVIEW = "basin-review"
BASIN_REPLAY = "basin-replay"
POINTWISE_REPLAY = "pointwise-replay"
TARGET_RHS = "target-rhs"
ARMS = ("proposer", "algorithm", "hybrid")
SEEDS = (100, 101)
BUDGET = {"max_evaluations": 1000, "wall_seconds": 180.0, "starts": 2}
SETTINGS = StagedModelSettings(
    model_revision=pilot.MODELS["20b"]["revision"],
    max_output_tokens=6144,
    maximum_requests=1,
    attempts_per_step=1,
    maximum_total_tokens=32768,
)


def _plan(payload):
    from scripts.submit_trajectory_functions import ENTRYPOINTS

    count = len(payload["graphs"]) * len(SEEDS)
    return {
        "protocol": PROTOCOL,
        "profile": payload["profile"],
        "intake_sha256": payload["artifact_sha256"],
        "settings": SETTINGS.model_dump(mode="json"),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "fit_budget": BUDGET,
        "maximum_fit_slots": len(ARMS) * count,
        "maximum_generation_requests": 0
        if payload["profile"] in (BASIN_REPLAY, POINTWISE_REPLAY)
        else 2 * count,
        "runtime_source": runtime_source_hash(),
        "entrypoints": {p: file_hash(pilot.REPO / p) for p in ENTRYPOINTS},
        "validation_identity": VALIDATION,
        "test_data_opened": False,
    }


def load(root: Path) -> tuple[dict, dict]:
    plan = sealed_read(root / "plan.json")
    payload = sealed_read(root / "intake.json")
    if plan["artifact_sha256"] != content_hash(_plan(payload)):
        raise ValueError("Prepared plan, source or training payload changed")
    return plan, payload


def training_case(data: dict, row: dict) -> dict:
    """Restore only the public training trajectories belonging to this graph."""
    case = data["cases"][row["case"]]
    return {
        **case,
        "evidence": {**case["evidence"], "trajectories": data["train"][row["dataset"]]},
    }


def read_tolerances(path: Path) -> dict:
    """Read a user-declared pointwise contract without overwriting repeated keys."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate pointwise tolerance key {key!r}")
            result[key] = value
        return result

    policy = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(policy, dict) or not policy:
        raise ValueError(
            "The pointwise tolerance file must contain a nonempty JSON object"
        )
    return policy


def checked_tolerances(src: dict, graphs: list, policy: dict) -> dict:
    """Bind each threshold to an observed differential target in this cohort."""
    targets: dict[str, set[str]] = {}
    for row in graphs:
        observed = src["cases"][row["case"]]["evidence"]["context"]["targets"]
        kinds = row["graph"]["variable_types"]
        targets.setdefault(row["dataset"], set()).update(
            t for t in observed if kinds.get(t) == "differential"
        )
    if not isinstance(policy, dict) or not policy:
        raise ValueError(
            "Provide a nonempty mapping from dataset IDs to target tolerances"
        )
    clean = {}
    for dataset, values in policy.items():
        if dataset not in targets or not isinstance(values, dict) or not values:
            raise ValueError(
                f"Expected a registered dataset and target tolerances: {dataset!r}"
            )
        clean[dataset] = {}
        for target, tolerance in values.items():
            if target not in targets[dataset]:
                raise ValueError(
                    f"Expected an observed differential target of {dataset}: {target}"
                )
            if (
                type(tolerance) not in (float, int)
                or not 0 <= tolerance <= sys.float_info.max
            ):
                raise ValueError(
                    f"Expected a finite nonnegative pointwise tolerance for {target}"
                )
            if float(tolerance) != tolerance:
                raise ValueError(
                    f"Pointwise tolerance for {target} loses precision"
                )
            clean[dataset][target] = float(tolerance)
    return clean


def prepare(
    root: Path, source: Path, profile: str = FIXED, *, pointwise_tolerances=None
):
    from autoformalism.research.preproposal_cases import BASIN
    from autoformalism.research.trajectory_function_families import build_shortlist
    from autoformalism.research.trajectory_function_fitting import validate_formula

    src = sealed_read(source)
    if src["artifact_sha256"] != TRAIN:
        raise ValueError("Expected the registered six-graph training-only cohort")
    if profile not in (FIXED, SOURCE_REVIEW, BASIN_REVIEW):
        raise ValueError("Unknown function-construction profile")
    graphs = [
        r for r in src["graphs"] if profile != BASIN_REVIEW or r["dataset"] == BASIN
    ]
    if not graphs:
        raise ValueError("No matching graphs in the registered intake")
    policy = (
        checked_tolerances(src, graphs, pointwise_tolerances)
        if pointwise_tolerances is not None
        else None
    )
    with _lock(root / "prepare-lock"):
        if (root / "plan.json").exists():
            plan, existing = load(root)
            if (
                plan["profile"] != profile
                or existing.get("pointwise_tolerances") != policy
            ):
                raise ValueError(
                    "This root belongs to a different profile or pointwise contract"
                )
            return plan
        menus = []
        for row in graphs:
            menu = build_shortlist(row["graph"], training_case(src, row))
            if len(menu["candidates"]) != 2:
                raise ValueError("Expected exactly two complete algorithm candidates")
            for form in menu["candidates"]:
                validate_formula(row["graph"], form)
            menus.append(menu)
        payload = sealed_write(
            root / "intake.json",
            {
                "source": TRAIN,
                "profile": profile,
                **({"pointwise_tolerances": policy} if policy is not None else {}),
                "graphs": graphs,
                "cases": {r["case"]: src["cases"][r["case"]] for r in graphs},
                "train": {r["dataset"]: src["train"][r["dataset"]] for r in graphs},
                "menus": menus,
                "observations": {
                    k: prompts.observations(src["train"][k])
                    for k in sorted({r["dataset"] for r in graphs})
                },
            },
        )
        return sealed_write(root / "plan.json", _plan(payload))


def slots(payload):
    """Give each graph two equally budgeted tickets in every arm."""
    return [
        (g, arm, rank)
        for g in range(len(payload["graphs"]))
        for arm in ARMS
        for rank in range(len(SEEDS))
    ]


def _request(payload, g, arm):
    r = payload["graphs"][g]
    interface = prompts if payload["profile"] == FIXED else revision_prompts
    system, packet, schema = interface.request(
        r["graph"],
        payload["cases"][r["case"]],
        payload["observations"][r["dataset"]],
        payload["menus"][g],
        arm == "hybrid",
    )
    tolerances = payload.get("pointwise_tolerances", {}).get(r["dataset"])
    if tolerances:
        packet["pointwise_accuracy_required"] = {
            "absolute_tolerances_in_target_units": tolerances,
            "rule": (
                "A proof that shared parameters cannot match the training observations "
                "within a listed tolerance prevents fitting."
            ),
        }
    return system, packet, schema


def _read_candidate(root, plan, i):
    path = root / "candidates" / f"{i:03}.json"
    result = sealed_read(path)
    if result["plan"] != plan["artifact_sha256"] or result["index"] != i:
        raise ValueError("Candidate checkpoint belongs to a different slot")
    if result.get("reply_sha256"):
        folder = root / "steps" / f"slot-{i:03}"
        reply = sealed_read(folder / "reply.json")
        if reply["artifact_sha256"] != result["reply_sha256"] or (
            "request_hash" in reply
            and file_hash(folder / "calls" / (reply["request_hash"] + ".json"))
            != reply["physical_receipt_sha256"]
        ):
            raise ValueError("Candidate's generation receipt changed")
    return result


def assess_reply(payload, g, arm, raw):
    """Keep fixed replies on their parent and bind approved edits to a child graph."""
    from autoformalism.research.trajectory_function_revision import assess_revision

    row = payload["graphs"][g]
    menu = payload["menus"][g]["candidates"]
    route, reason = "fixed_graph", None
    if payload["profile"] == FIXED:
        formula, origin = prompts.to_formula(raw, menu)
    else:
        route, formula, origin, reason = revision_prompts.decode_reply(
            row["graph"], raw, menu, hybrid=arm == "hybrid"
        )
    if route == "request_topology_change":
        review = assess_revision(
            row["graph"], training_case(payload, row), formula, reason
        )
        # The parent hash stays on the slot. Only an admitted child is executable.
        fields = {k: v for k, v in review.items() if k not in {"formula", "graph"}}
        record = {
            "route": route,
            "scientific_reason": reason,
            "origin": origin,
            **fields,
        }
        if review["status"] == "eligible":
            record.update(
                formula=review["formula"],
                revised_graph=review["graph"],
                revised_graph_sha256=content_hash(review["graph"]),
            )
        return record
    return fixed_candidate(payload, row, formula, origin)


def fixed_candidate(payload, row, formula, origin):
    """Record a fixed formula, keeping the parent identifier owned by the slot."""
    from autoformalism.research.trajectory_function_fitting import validate_formula
    from autoformalism.research.trajectory_function_revision import assess_fixed

    route = "fixed_graph"
    if payload["profile"] != FIXED:
        review = assess_fixed(row["graph"], training_case(payload, row), formula)
        return {
            "route": route,
            "origin": origin,
            **{k: v for k, v in review.items() if k != "graph"},
        }
    try:
        validate_formula(row["graph"], formula)
    except (ValueError, ModelValidationError) as exc:
        return {"status": "invalid_formula", "route": route, "reason": str(exc)}
    return {"status": "eligible", "route": route, "formula": formula, "origin": origin}


def algorithm_candidate(data, g, rank):
    """Apply the same formula checks to the two algorithmic reference models."""
    form = data["menus"][g]["candidates"][rank]
    return fixed_candidate(data, data["graphs"][g], form, "ranked_family")


def fitting_graph(candidate, row):
    """Resolve the graph attached to this candidate before fitting or scoring."""
    if candidate["graph"] != row["structure_sha256"]:
        raise ValueError("Candidate parent graph differs from its slot")
    if candidate.get("route") == "request_topology_change":
        graph = candidate["revised_graph"]
        if content_hash(graph) != candidate["revised_graph_sha256"]:
            raise ValueError("Revised graph differs from its admission")
        return graph
    return row["graph"]


def pointwise_admission(data: dict, row: dict, candidate: dict) -> dict:
    """Check both function-level proofs for each declared pointwise tolerance."""
    policy = data.get("pointwise_tolerances", {}).get(row["dataset"])
    if candidate["status"] != "eligible" or not policy:
        return candidate
    from autoformalism.research.affine_direction_certificate import certify
    from autoformalism.research.ordered_driver_certificate import (
        certify as certify_order,
    )
    from autoformalism.search.topology_compatibility import (
        TrainingCompatibilityEvidence,
    )

    graph = fitting_graph(candidate, row)
    case = training_case(data, row)
    evidence = TrainingCompatibilityEvidence.model_validate(case["evidence"])
    checks = {}
    ordered_checks = {}
    feedback = []
    for target, tolerance in sorted(policy.items()):
        proof = certify(
            graph, candidate["formula"], target, evidence, tolerance=tolerance
        )
        checks[target] = proof
        if proof["status"] == "conflict":
            feedback.append(_direction_feedback(data, graph, case, proof))
        order = certify_order(
            graph, candidate["formula"], target, evidence, tolerance=tolerance
        )
        ordered_checks[target] = order
        if order["status"] == "conflict":
            feedback.append(
                {
                    "check": "ordered_driver",
                    "target": target,
                    "absolute_pointwise_tolerance": tolerance,
                    "rhs": candidate["formula"]["equations"][target],
                    "driver": order["driver"],
                    "worst_sampled_absolute_error_lower_bound": order[
                        "worst_sampled_absolute_error_lower_bound"
                    ],
                    "witnesses": order["witnesses"],
                    "message": (
                        f"Within each pair, {target} starts at the same measured "
                        "value and the other forcing histories agree through the "
                        f"listed time. Higher {order['driver']} throughout gives "
                        f"opposite observed orderings of {target}. No shared "
                        "linear driver gain can match both within the tolerance."
                    ),
                    **_repair_options(data, graph, case, target, order["driver"]),
                }
            )
    result = {**candidate, "pointwise_checks": checks, "ordered_checks": ordered_checks}
    if feedback:
        result.update(status="pointwise_conflict", repair_feedback=feedback)
    return result


def _direction_feedback(data, graph, case, proof):
    """Show the witness intervals and permissible repairs."""
    target, driver = proof["target"], proof["driver"]
    ref = proof["reference"]
    intervals = {
        kind: {
            key: proof[kind][key]
            for key in (
                "event",
                "time",
                "driver_range",
                "observed",
                "required_change",
                "maximum_permitted_change",
            )
        }
        for kind in ("rise", "fall")
    }
    feedback = {
        "target": target,
        "absolute_pointwise_tolerance": proof["tolerance"],
        "rhs": proof["raw_rhs"],
        "depletion_gate_scale": proof["gate_threshold"],
        "driver": driver,
        "driver_side": proof["side"],
        "message": (
            f"The observed rise and fall of {target} rule out both slope signs "
            "in this affine rate. Refitting its coefficients cannot satisfy "
            "this pointwise requirement."
        ),
        "reference": {
            key: ref[key]
            for key in (
                "event",
                "time",
                "forcing_values",
                "initial_target",
                "observed_final",
                "derivative_bound",
            )
        },
        **intervals,
        "assumptions": [
            "Shared parameters, observed initial target, complete piecewise-linear "
            "exact forcing or fixed covariates, open-loop dynamics, exhaustive "
            "dependencies and no resets.",
            *ref["assumptions"],
        ],
        "target_observation_uncertainty": proof["observation_uncertainty"],
        **_repair_options(data, graph, case, target, driver),
    }
    return feedback


def _repair_options(data, graph, case, target, driver):
    """Describe source changes in the edit interface for this campaign."""
    options = [
        f"Revise the rate law for {target} on its declared sources. "
        f"A nonlinear response to {driver} is one possibility."
    ]
    feedback = {"current_sources": graph["sources"][target], "repair_options": options}
    profile = data.get("source_profile", data["profile"])
    if profile != FIXED:
        from autoformalism.research.trajectory_function_revision import (
            permitted_sources,
        )

        feedback["sources_available_to_add"] = sorted(
            set(permitted_sources(graph, case)) - set(graph["sources"][target])
        )
        if feedback["sources_available_to_add"]:
            options.append(
                f"Include another permitted source directly in the revised {target} "
                "rate if scientifically justified."
                if profile == TARGET_RHS
                else "Use request_topology_change if a scientifically justified rate "
                "depends on another available source."
            )
    return feedback


def run(root, endpoint, *, runtime_record=None, transport=None, token_transport=None):
    from scripts import submit_t2_breadth

    plan, p = load(root)
    if p["profile"] in (BASIN_REPLAY, POINTWISE_REPLAY):
        raise ValueError("Saved reply replay has no generation allowance")
    if transport is None:
        if runtime_record is None:
            raise ValueError("Live generation requires the sealed 20B runtime record")
        submit_t2_breadth.bind(root, Path(runtime_record), endpoint)
    with _lock(root / "generation-lock"):
        for i, (g, arm, rank) in enumerate(slots(p)):
            path = root / "candidates" / f"{i:03}.json"
            if path.exists():
                _read_candidate(root, plan, i)
                continue
            row = p["graphs"][g]
            record = {
                "index": i,
                "graph": row["structure_sha256"],
                "arm": arm,
                "rank": rank,
                "plan": plan["artifact_sha256"],
                "seed": SEEDS[rank],
            }
            if arm == "algorithm":
                record.update(algorithm_candidate(p, g, rank))
            else:
                system, user, schema = _request(p, g, arm)
                reply = pilot._ask(
                    root,
                    plan,
                    f"slot-{i:03}",
                    SEEDS[rank],
                    system,
                    user,
                    schema,
                    endpoint,
                    transport,
                    token_transport,
                    {"physical_requests": 0, "slots": {}},
                    plan["maximum_generation_requests"],
                    sampling_step=f"graph-{g}-functions",
                )
                record.update(
                    status=reply["status"], reply_sha256=reply["artifact_sha256"]
                )
                if reply["status"] == "visible":
                    try:
                        raw = schema.model_validate(reply["value"]).model_dump()
                        record.update(assess_reply(p, g, arm, raw))
                    except (ValueError, ModelValidationError) as exc:
                        record.update(status="invalid_formula", reason=str(exc))
                elif reply.get("error"):
                    record["reason"] = reply["error"]
            sealed_write(path, pointwise_admission(p, row, record))
        identities = [
            _read_candidate(root, plan, i)["artifact_sha256"]
            for i in range(len(slots(p)))
        ]
        sealed_write(
            root / "generation.json",
            {"plan": plan["artifact_sha256"], "candidates": identities},
        )
    return status(root)


def _frozen_candidates(root, plan):
    frozen = sealed_read(root / "generation.json")
    if (
        frozen["plan"] != plan["artifact_sha256"]
        or len(frozen["candidates"]) != plan["maximum_fit_slots"]
    ):
        raise ValueError("Generation freeze differs")
    return frozen


def fit(root, index):
    """Consume one fit allowance only after the common candidate freeze."""
    from autoformalism.research.trajectory_function_fitting import fit_formula

    plan, p = load(root)
    jobs = slots(p)
    if index < 0 or index >= len(jobs):
        raise ValueError("Fit index out of range")
    frozen = _frozen_candidates(root, plan)
    candidate = _read_candidate(root, plan, index)
    if candidate["artifact_sha256"] != frozen["candidates"][index]:
        raise ValueError("Candidate differs from generation freeze")
    g, _, rank = jobs[index]
    row = p["graphs"][g]
    directory = root / "fits" / f"{index:03}"
    identity = {
        "index": index,
        "candidate": candidate["artifact_sha256"],
        "generation": frozen["artifact_sha256"],
    }
    path = directory / "result.json"
    with _lock(directory):
        if path.exists():
            record = sealed_read(path)
            if any(record[k] != v for k, v in identity.items()):
                raise ValueError("Fit checkpoint differs from its candidate")
            return record
        if candidate["status"] != "eligible":
            result = {"status": "ineligible", "reason": candidate["status"]}
        elif p["profile"] == POINTWISE_REPLAY:
            raise ValueError("Admission-only replay has no optimizer allowance")
        elif (directory / "reservation.json").exists():
            if (
                sealed_read(directory / "reservation.json")["candidate"]
                != identity["candidate"]
            ):
                raise ValueError("Changed fitting reservation")
            result = {
                "status": "interrupted",
                "reason": "Previous fit consumed its reserved allowance",
            }
        else:
            sealed_write(directory / "reservation.json", identity)
            result = fit_formula(
                fitting_graph(candidate, row),
                candidate["formula"],
                p["train"][row["dataset"]],
                seed=SEEDS[rank],
                **plan["fit_budget"],
            )
        return sealed_write(path, {**identity, "result": result})


def finalists(root, plan, p):
    """Freeze train-selected models for every graph/arm before reading validation."""
    frozen = _frozen_candidates(root, plan)
    fits = []
    groups = {}
    for i, (g, arm, _) in enumerate(slots(p)):
        r = sealed_read(root / "fits" / f"{i:03}" / "result.json")
        if (
            r["index"] != i
            or r["candidate"] != frozen["candidates"][i]
            or r["generation"] != frozen["artifact_sha256"]
        ):
            raise ValueError("Fit and candidate freeze differ")
        fits.append(r)
        group = groups.setdefault((g, arm), [])
        outcome = r["result"]
        if outcome["status"] == "fitted" and math.isfinite(outcome["training_nmse"]):
            group.append((outcome["training_nmse"], i))
    winners = [
        {"graph_index": g, "arm": arm, "winner": min(group)[1] if group else None}
        for (g, arm), group in groups.items()
    ]
    return sealed_write(
        root / "finalists.json",
        {
            "plan": plan["artifact_sha256"],
            "fit_receipts": [r["artifact_sha256"] for r in fits],
            "winners": winners,
        },
    ), fits


def validate(root, validation: Path):
    from autoformalism.research.trajectory_function_fitting import score_formula

    plan, p = load(root)
    with _lock(root / "validation-lock"):
        frozen, fits = finalists(root, plan, p)
        saved = sealed_read(validation)
        if (
            saved["artifact_sha256"] != plan["validation_identity"]
            or saved["training_intake_sha256"] != p["source"]
        ):
            raise ValueError("Validation intake differs from registered public release")
        decisions = []
        for j, item in enumerate(frozen["winners"]):
            path = root / "validation" / f"{j:03}.json"
            i = item["winner"]
            row = p["graphs"][item["graph_index"]]
            if path.exists():
                decision = sealed_read(path)
                if (
                    decision["finalists"] != frozen["artifact_sha256"]
                    or decision["selection"] != item
                ):
                    raise ValueError("Validation checkpoint differs")
            else:
                score = {"status": "no_verified_fit"}
                if i is not None:
                    candidate = _read_candidate(root, plan, i)
                    if candidate["artifact_sha256"] != fits[i]["candidate"]:
                        raise ValueError("Selected formula differs")
                    score = score_formula(
                        fitting_graph(candidate, row),
                        candidate["formula"],
                        fits[i]["result"],
                        saved["rows"][row["dataset"]],
                    )
                decision = sealed_write(
                    path,
                    {
                        "selection": item,
                        "finalists": frozen["artifact_sha256"],
                        "score": score,
                    },
                )
            decisions.append(decision["artifact_sha256"])
        sealed_write(
            root / "evaluation.json",
            {
                "finalists": frozen["artifact_sha256"],
                "validation": saved["artifact_sha256"],
                "results": decisions,
            },
        )
    return status(root)


def status(root):
    plan, p = load(root)
    generated = [
        _read_candidate(root, plan, i)
        for i in range(len(slots(p)))
        if (root / "candidates" / f"{i:03}.json").exists()
    ]
    total = plan["maximum_fit_slots"]
    generation = None
    if (root / "generation.json").is_file():
        generation = _frozen_candidates(root, plan)
        if [row["artifact_sha256"] for row in generated] != generation["candidates"]:
            raise ValueError("Generation manifest differs from candidate receipts")
    fits = []
    for i in range(total):
        path = root / "fits" / f"{i:03}" / "result.json"
        if path.exists():
            row = sealed_read(path)
            if (
                not generation
                or row["index"] != i
                or row["candidate"] != generation["candidates"][i]
                or row["generation"] != generation["artifact_sha256"]
            ):
                raise ValueError("Fit receipt differs from generation")
            fits.append(row)
    evaluated = False
    decisions = []
    if (root / "evaluation.json").is_file():
        freeze = sealed_read(root / "finalists.json")
        evaluation = sealed_read(root / "evaluation.json")
        if (
            freeze["plan"] != plan["artifact_sha256"]
            or len(fits) != total
            or freeze["fit_receipts"] != [r["artifact_sha256"] for r in fits]
            or evaluation["finalists"] != freeze["artifact_sha256"]
            or evaluation["validation"] != plan["validation_identity"]
            or len(freeze["winners"]) != len(p["graphs"]) * len(ARMS)
            or len(evaluation["results"]) != len(freeze["winners"])
        ):
            raise ValueError("Evaluation manifest differs from the frozen fits")
        for j, item in enumerate(freeze["winners"]):
            path = root / "validation" / f"{j:03}.json"
            if not path.is_file():
                continue
            row = sealed_read(path)
            if (
                row["artifact_sha256"] != evaluation["results"][j]
                or row["finalists"] != freeze["artifact_sha256"]
                or row["selection"] != item
            ):
                raise ValueError("Validation receipt differs from the frozen selection")
            decisions.append(j)
        evaluated = len(decisions) == len(freeze["winners"])
    return {
        "status": "complete" if evaluated else "incomplete",
        "generation_complete": generation is not None,
        "cases": len(p["graphs"]),
        "candidate_slots": len(slots(p)),
        "recorded": len(generated),
        "candidate_statuses": dict(Counter(r["status"] for r in generated)),
        "physical_requests": len(list((root / "steps").glob("*/calls/*.json"))),
        "fit_receipts": len(fits),
        "fit_statuses": dict(Counter(r["result"]["status"] for r in fits)),
        "validation_records": len(list((root / "validation").glob("*.json"))),
        "test_data_opened": False,
    }


def preflight(root):
    from autoformalism.research.trajectory_function_fitting import validate_formula
    from scripts.trajectory_function_report import prompt_review

    plan, p = load(root)
    if p["profile"] == POINTWISE_REPLAY:
        raise ValueError("Admission-only replay has no optimizer allowance")
    for i, r in enumerate(p["graphs"]):
        for arm in ("hybrid", "proposer"):
            *_, schema = _request(p, i, arm)
            strict_provider_schema(schema.model_json_schema())
        for f in p["menus"][i]["candidates"]:
            validate_formula(r["graph"], f)
    from autoformalism.research.trajectory_function_fitting import fit_formula

    # Exercise the native solver with a training trajectory before queueing GPUs.
    example = (p["graphs"][0], p["menus"][0]["candidates"][0])
    if "pointwise_tolerances" in p:
        example = _admitted_smoke_example(p)
    smoke_status = "skipped_no_admitted_reference"
    if example is not None:
        graph, formula = example
        smoke = fit_formula(
            graph["graph"],
            formula,
            p["train"][graph["dataset"]][:1],
            max_evaluations=1,
            starts=1,
            wall_seconds=30.0,
        )
        smoke_status = smoke["status"]
        if smoke_status != "fitted":
            raise ValueError(f"Training rollout preflight failed: {smoke_status}")
    prompt_review(root, p, _request)
    return {
        "status": "preflight_passed",
        "training_rollout_smoke": smoke_status,
        "graphs": len(p["graphs"]),
        "generation_budget": plan["maximum_generation_requests"],
        "fit_slots": plan["maximum_fit_slots"],
        "fit_budget": BUDGET,
    }


def _admitted_smoke_example(data):
    for g, row in enumerate(data["graphs"]):
        for rank in range(len(data["menus"][g]["candidates"])):
            candidate = algorithm_candidate(data, g, rank)
            candidate["graph"] = row["structure_sha256"]
            if pointwise_admission(data, row, candidate)["status"] == "eligible":
                return row, candidate["formula"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ["prepare", "run", "fit", "validate", "status", "inspect", "preflight"]:
        c = commands.add_parser(name)
        c.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            c.add_argument("--source", type=Path, required=True)
            c.add_argument(
                "--pointwise-tolerances",
                type=Path,
                help="JSON mapping dataset IDs to target names and absolute tolerances",
            )
            c.add_argument(
                "--profile", choices=(FIXED, SOURCE_REVIEW, BASIN_REVIEW), default=FIXED
            )
        if name == "run":
            c.add_argument("--endpoint", required=True)
            c.add_argument("--runtime-record", type=Path, required=True)
        if name == "fit":
            c.add_argument("--index", type=int, required=True)
        if name == "validate":
            c.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        policy = (
            read_tolerances(args.pointwise_tolerances)
            if args.pointwise_tolerances
            else None
        )
        plan = prepare(root, args.source, args.profile, pointwise_tolerances=policy)
        result = {
            "plan": plan["artifact_sha256"],
            "fit_slots": plan["maximum_fit_slots"],
        }
    elif args.action == "run":
        result = run(root, args.endpoint, runtime_record=args.runtime_record)
    elif args.action == "fit":
        checkpoint = fit(root, args.index)
        result = {"index": args.index, "status": checkpoint["result"]["status"]}
    elif args.action == "validate":
        result = validate(root, args.validation)
    elif args.action == "status":
        result = status(root)
    elif args.action == "inspect":
        from scripts.trajectory_function_report import inspect

        result = inspect(root, status(root))
    else:
        result = preflight(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
