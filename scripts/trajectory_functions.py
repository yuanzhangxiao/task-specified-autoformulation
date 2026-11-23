"""Compare 20B, trajectory-ranked functions and their combination on fixed graphs."""

from __future__ import annotations

import argparse
import json
import math
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
from scripts.construction_functions import file_hash

PROTOCOL = "trajectory-function-comparison-1"
TRAIN = "7e4bb4712fdf778b0174a4945f89f9f30a8d9c9fcc648591096db03c60fa4ecd"
VALIDATION = "bfd00cc19e3194056166119c634576a3278382b50d13636e6687a8321a53801d"
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
        "intake_sha256": payload["artifact_sha256"],
        "settings": SETTINGS.model_dump(mode="json"),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "fit_budget": BUDGET,
        "maximum_fit_slots": len(ARMS) * count,
        "maximum_generation_requests": 2 * count,
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


def prepare(root: Path, source: Path):
    from autoformalism.research.trajectory_function_families import build_shortlist
    from autoformalism.research.trajectory_function_fitting import validate_formula

    src = sealed_read(source)
    if src["artifact_sha256"] != TRAIN:
        raise ValueError("Expected the registered six-graph training-only cohort")
    with _lock(root / "prepare-lock"):
        if (root / "plan.json").exists():
            return load(root)[0]
        menus = []
        for row in src["graphs"]:
            case = src["cases"][row["case"]]
            frozen = {
                **case,
                "evidence": {
                    **case["evidence"],
                    "trajectories": src["train"][row["dataset"]],
                },
            }
            menu = build_shortlist(row["graph"], frozen)
            if len(menu["candidates"]) != 2:
                raise ValueError("Expected exactly two complete algorithm candidates")
            for form in menu["candidates"]:
                validate_formula(row["graph"], form)
            menus.append(menu)
        payload = sealed_write(
            root / "intake.json",
            {
                "source": TRAIN,
                "graphs": src["graphs"],
                "cases": src["cases"],
                "train": src["train"],
                "menus": menus,
                "observations": {
                    k: prompts.observations(v) for k, v in src["train"].items()
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
    return prompts.request(
        r["graph"],
        payload["cases"][r["case"]],
        payload["observations"][r["dataset"]],
        payload["menus"][g],
        arm == "hybrid",
    )


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


def run(root, endpoint, *, runtime_record=None, transport=None, token_transport=None):
    from autoformalism.research.trajectory_function_fitting import validate_formula
    from scripts import submit_t2_breadth

    plan, p = load(root)
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
            menu = p["menus"][g]["candidates"]
            if arm == "algorithm":
                record.update(
                    status="eligible", formula=menu[rank], origin="ranked_family"
                )
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
                        form, origin = prompts.to_formula(raw, menu)
                        validate_formula(row["graph"], form)
                    except (ValueError, ModelValidationError) as exc:
                        record.update(status="invalid_formula", reason=str(exc))
                    else:
                        record.update(status="eligible", formula=form, origin=origin)
                elif reply.get("error"):
                    record["reason"] = reply["error"]
            sealed_write(path, record)
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
                row["graph"],
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
                        row["graph"],
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
    for i, r in enumerate(p["graphs"]):
        for arm in ("hybrid", "proposer"):
            *_, schema = _request(p, i, arm)
            strict_provider_schema(schema.model_json_schema())
        for f in p["menus"][i]["candidates"]:
            validate_formula(r["graph"], f)
    from autoformalism.research.trajectory_function_fitting import fit_formula

    # Exercise the native solver with a training trajectory before queueing GPUs.
    graph = p["graphs"][0]
    smoke = fit_formula(
        graph["graph"],
        p["menus"][0]["candidates"][0],
        p["train"][graph["dataset"]][:1],
        max_evaluations=1,
        starts=1,
        wall_seconds=30.0,
    )
    if smoke["status"] != "fitted":
        raise ValueError(f"Training rollout preflight failed: {smoke['status']}")
    prompt_review(root, p, _request)
    return {
        "status": "preflight_passed",
        "training_rollout_smoke": smoke["status"],
        "graphs": len(p["graphs"]),
        "generation_budget": plan["maximum_generation_requests"],
        "fit_slots": plan["maximum_fit_slots"],
        "fit_budget": BUDGET,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ["prepare", "run", "fit", "validate", "status", "inspect", "preflight"]:
        c = commands.add_parser(name)
        c.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            c.add_argument("--source", type=Path, required=True)
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
        plan = prepare(root, args.source)
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
