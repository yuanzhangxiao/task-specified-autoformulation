"""Compare the six frozen basin process replies under a common starting-value policy."""

from __future__ import annotations

import argparse
import ast
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from autoformalism.fitting.public_fitting import _lock
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.trajectory_function_fitting import (
    _Rollout,
    _rows,
    score_formula,
)
from autoformalism.staged_topology import content_hash
from scripts import basin_function_family as family
from scripts import basin_process_proposals as proposals
from scripts import fit_basin_process_proposals as fitting
from scripts import replay_basin_process_proposals as replay
from scripts import trajectory_functions as trial
from scripts.construction_functions import file_hash

SOURCES = {
    "full": (
        "7a04e2664950cac6f5d180cf3fda3575a54dfda66c5f019d5b7ce44948542312",
        "3eb165b7b30779867c9b14bb057bac3c1eb8d1081aa6dae605808e012afaeed7",
        proposals.ARCHIVED_REQUEST,
    ),
    "control": (
        "203fee1c1c0526573771ae7ee7ad05052d6e8bb7a00defa76e610bff96db3560",
        "d676de00478e38c81789eea0610aa492d9f3b33ccb17f750bed788e3135dee1f",
        "1e4bceabead72ae9d715e4d5d18da55a4bffa353fd9f83ea2b513cd21e2bbd01",
    ),
}
STARTS = (1.0, 10.0, 100.0)
BUDGET = {**trial.BUDGET, "seed": 100}
FAMILIES = {
    (100, "full"): "Independent gains multiplying head^(3/2), redundant zero clamp",
    (100, "control"): "Independent gains multiplying head^(3/2)",
    (101, "full"): "One shared gain multiplying head*(1+sqrt(head))",
    (101, "control"): "Independent gains multiplying head^2",
    (102, "full"): "Literal 55 multiplying head^(5/2) on both sides",
    (102, "control"): "Literal 0.5 upstream and 0.3 downstream multiplying head^(3/2)",
}


def _code() -> dict:
    return {
        "driver": file_hash(Path(__file__)),
        "fit_admission": file_hash(Path(fitting.__file__)),
        "assembly": file_hash(Path(proposals.__file__)),
        **family._code(),
    }


def read_source(root: Path, profile: str) -> tuple:
    plan = sealed_read(root / "plan.json")
    intake = sealed_read(root / "intake.json")
    frozen = trial._frozen_candidates(root, plan)
    prompt = sealed_read(
        root / ("original" if profile == "full" else ".") / "prompt.json"
    )
    plan_sha, gen_sha, request = SOURCES[profile]
    if (
        plan["artifact_sha256"] != plan_sha
        or frozen["artifact_sha256"] != gen_sha
        or plan["intake_sha256"] != intake["artifact_sha256"]
        or intake["artifact_sha256"] != family.INTAKE
        or intake["source"] != trial.TRAIN
        or plan["parent"] != family.PARENT
        or plan["seeds"] != list(proposals.SEEDS)
        or content_hash([prompt[k] for k in ("system", "user", "schema")]) != request
    ):
        raise ValueError("Source differs from its registered prompt and generation")
    if profile == "full":
        original = sealed_read(root / "original" / "plan.json")
        if (
            plan["source_plan"] != replay.SOURCE_PLAN
            or original["request_sha256"] != request
        ):
            raise ValueError("Wrong source replay")
    elif (
        plan.get("evidence_profile") != proposals.OMIT_OUTLET_ROWS
        or plan["comparison_full_request"] != SOURCES["full"][2]
    ):
        raise ValueError("Expected live prompt with outlet rows withheld")
    candidates = [
        trial._read_candidate(root, plan, i) for i in range(len(plan["seeds"]))
    ]
    for seed, c, digest in zip(
        plan["seeds"], candidates, frozen["candidates"], strict=True
    ):
        if (
            c["artifact_sha256"] != digest
            or c["seed"] != seed
            or c["status"] != "eligible"
            or c["graph"] != family.PARENT
            or content_hash(c["revised_graph"]) != family.REVISED
        ):
            raise ValueError("Candidate lost its registered identity or graph")
    if profile == "full":
        replay.verify_provenance(root, candidates)
    return intake, prompt, candidates


def prepare(full: Path, control: Path, root: Path) -> tuple[dict, list]:
    if any(root.resolve().is_relative_to(p.resolve()) for p in (full, control)):
        raise ValueError("Use an output root separate from both archived experiments")
    sources = {
        profile: read_source(path, profile)
        for profile, path in (("full", full), ("control", control))
    }
    data, prompt, _ = sources["full"]
    edited = deepcopy(prompt["user"])
    if not edited["outlet_evidence"]["rows"]:
        raise ValueError("Full prompt missing its outlet evidence")
    edited["outlet_evidence"]["rows"] = []
    other = sources["control"][1]
    if edited != other["user"] or any(
        prompt[k] != other[k] for k in ("system", "schema")
    ):
        raise ValueError("Unexpected prompt difference")
    designs = []
    models = []
    for profile, (_, _, candidates) in sources.items():
        for c in candidates:
            key = f"{profile}-{c['seed']}"
            params = c["formula"]["parameters"]
            designs.append(
                {
                    "key": key,
                    "seed": c["seed"],
                    "profile": profile,
                    "candidate": c["artifact_sha256"],
                    "authored_factors": c["authored_factors"],
                    "revised_graph": c["revised_graph"],
                    "submitted_starts": params,
                    "family": FAMILIES[c["seed"], profile],
                }
            )
            for start in STARTS if params else (None,):
                f = (
                    c["formula"]
                    if start is None
                    else c["formula"] | {"parameters": dict.fromkeys(params, start)}
                )
                models.append({"key": key, "start": start, "formula": f})
    return sealed_write(
        root / "plan.json",
        {
            "protocol": "basin-process-common-start-1",
            "sources": {p: list(v) for p, v in SOURCES.items()},
            "code": _code(),
            "training_intake": trial.TRAIN,
            "validation_intake": trial.VALIDATION,
            "budget": BUDGET,
            "fixed_budget": {**BUDGET, "max_evaluations": 1, "starts": 1},
            "graph": designs[0]["revised_graph"],
            "designs": designs,
            "models": models,
            "selection": "Rank physically qualified starts by training NMSE per seed.",
            "cross_check": (
                "At both selected seed-100 vectors, compare upstream and downstream "
                "predictions by event without refitting."
            ),
            "test_data_opened": False,
        },
    ), data["train"][proposals.BASIN]


def _factor_gains(design):
    """Find a single independent fitted gain in each published seed-100 factor."""
    gains = set(design["submitted_starts"])
    ordered = []
    for side in ("upstream", "downstream"):
        tree = ast.parse(design["authored_factors"][f"{side}_factor"], mode="eval")
        live = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & gains
        if len(live) != 1:
            raise ValueError("Need one fitted gain per factor to cross-transfer")
        ordered.append(live.pop())
    if len(set(ordered)) != 2 or set(ordered) != gains:
        raise ValueError("Expected distinct upstream and downstream gains")
    return ordered


def cross_rollouts(
    plan: dict, ranking: dict, fits: list, rows: list, split: str
) -> dict:
    """Replay both donor vectors through the paired seed-100 models."""
    parsed = _rows(rows, split)
    designs = {d["key"]: d for d in plan["designs"]}
    keys = ("full-100", "control-100")
    winners = ranking["selected"]
    if any(winners[k] is None for k in keys):
        return {"status": "unavailable", "reason": "Missing qualified donor"}
    systems = {}
    for k in keys:
        m = plan["models"][winners[k]]
        engine = _Rollout(plan["graph"], m["formula"], ["h_down"], 4)
        systems[k] = (
            engine,
            [engine.record(r) for r in parsed],
            _factor_gains(designs[k]),
        )
    cases = []
    for donor in keys:
        fit = fits[winners[donor]]["result"]
        names = systems[donor][2]
        gains = [fit["parameters"][n] for n in names]
        outputs = {}
        for key in keys:
            engine, events, target_names = systems[key]
            params = dict(zip(target_names, gains, strict=True))
            outputs[key] = engine.run(events, [params[n] for n in engine.parameters])
        comparisons = []
        for row, (_, a), (_, b) in zip(
            parsed, outputs[keys[0]], outputs[keys[1]], strict=True
        ):
            states = systems[keys[0]][0].states
            comparisons.append(
                {
                    "event": row.trajectory_id,
                    "max_abs_m": {
                        name: float(np.max(abs(a[j] - b[j])))
                        for j, name in enumerate(states)
                    },
                }
            )
        cases.append(
            {
                "donor": donor,
                "upstream_gain": gains[0],
                "downstream_gain": gains[1],
                "events": comparisons,
            }
        )
    maximum = max(
        v for c in cases for e in c["events"] for v in e["max_abs_m"].values()
    )
    return {
        "status": "agreement" if maximum <= 1e-10 else "different",
        "max_abs_m": maximum,
        "tolerance_m": 1e-10,
        "vectors": cases,
    }


def train(full: Path, control: Path, root: Path) -> dict:
    """Fit under common or fixed starts and freeze six within-seed selections."""
    with _lock(root):
        plan, rows = prepare(full, control, root)
        designs = {d["key"]: d for d in plan["designs"]}
        fits = []
        qualified = {k: [] for k in designs}
        failures = []
        for i, m in enumerate(plan["models"]):
            budget = plan["budget"] if m["start"] is not None else plan["fixed_budget"]
            record = family._fit(root, plan | {"budget": budget}, i, rows)
            fits.append(record)
            r = record["result"]
            errors = fitting.fitted_checks(designs[m["key"]], r)
            failures.append(errors)
            if not errors:
                qualified[m["key"]].append((r["training_nmse"], i))
        ranking = sealed_write(
            root / "ranking.json",
            {
                "plan": plan["artifact_sha256"],
                "fit_receipts": [f["artifact_sha256"] for f in fits],
                "failures": failures,
                "selected": {k: min(v)[1] if v else None for k, v in qualified.items()},
                "total_evaluations": sum(
                    f["result"].get("evaluations", 0) for f in fits
                ),
                "validation_opened": False,
            },
        )
        cross = cross_rollouts(plan, ranking, fits, rows, "train")
        sealed_write(
            root / "train-cross.json", {"ranking": ranking["artifact_sha256"], **cross}
        )
        return ranking


def validate(root: Path, validation: Path) -> dict:
    """Score each frozen seed under its own selected vector, then cross-transfer."""
    with _lock(root):
        plan = sealed_read(root / "plan.json")
        ranks = sealed_read(root / "ranking.json")
        if (
            ranks["plan"] != plan["artifact_sha256"]
            or plan["code"] != _code()
            or len(ranks["fit_receipts"]) != len(plan["models"])
        ):
            raise ValueError("Incorrect training selection")
        fits = []
        for i, digest in enumerate(ranks["fit_receipts"]):
            r = sealed_read(root / "fits" / f"{i:02}" / "result.json")
            if (
                r["artifact_sha256"] != digest
                or r["slot"] != i
                or r["plan"] != ranks["plan"]
            ):
                raise ValueError("Candidate, fit or ranking changed")
            fits.append(r)
        if validation.name != "validation.json":
            raise ValueError("Expected validation.json")
        val = sealed_read(validation)
        if (
            val["artifact_sha256"] != plan["validation_intake"]
            or val["training_intake_sha256"] != plan["training_intake"]
        ):
            raise ValueError("Incorrect public validation intake")
        rows = val["rows"][proposals.BASIN]
        summaries = []
        for key, i in ranks["selected"].items():
            path = root / "validation" / f"{key}.json"
            identity = {"ranking": ranks["artifact_sha256"], "key": key, "index": i}
            if path.exists():
                record = sealed_read(path)
                if any(record[k] != v for k, v in identity.items()):
                    raise ValueError("Validation checkpoint differs")
            else:
                score = {"status": "no_qualified_fit", "pooled_nmse": None}
                if i is not None:
                    score = score_formula(
                        plan["graph"],
                        plan["models"][i]["formula"],
                        fits[i]["result"],
                        rows,
                    )
                record = sealed_write(path, identity | {"score": score})
            selected = fits[i]["result"] if i is not None else {}
            family_rows = [
                {
                    "start": m["start"],
                    "training_nmse": f["result"].get("training_nmse"),
                    "evaluations": f["result"].get("evaluations"),
                    "failures": e,
                }
                for m, f, e in zip(plan["models"], fits, ranks["failures"], strict=True)
                if m["key"] == key
            ]
            summaries.append(
                {
                    "key": key,
                    "selected": i,
                    "candidates": family_rows,
                    "selected_start": plan["models"][i]["start"]
                    if i is not None
                    else None,
                    "training_nmse": selected.get("training_nmse"),
                    "fitted_parameters": selected.get("parameters"),
                    "validation_nmse": record["score"]["pooled_nmse"],
                    "validation_status": record["score"]["status"],
                    "receipt": record["artifact_sha256"],
                }
            )
        cross = sealed_write(
            root / "val-cross.json",
            {
                "ranking": ranks["artifact_sha256"],
                **cross_rollouts(plan, ranks, fits, rows, "val"),
            },
        )
        return sealed_write(
            root / "summary.json",
            {
                "ranking": ranks["artifact_sha256"],
                "rows": summaries,
                "cross_check": cross["artifact_sha256"],
                "max_cross_difference_m": cross.get("max_abs_m"),
                "total_evaluations": ranks["total_evaluations"],
                "llm_calls": 0,
                "test_data_opened": False,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage, flags in (("train", ("full", "control")), ("validate", ("validation",))):
        p = commands.add_parser(stage)
        p.add_argument("--root", required=True, type=Path)
        for flag in flags:
            p.add_argument("--" + flag, required=True, type=Path)
    a = parser.parse_args()
    value = (
        train(a.full, a.control, a.root)
        if a.stage == "train"
        else validate(a.root, a.validation)
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
