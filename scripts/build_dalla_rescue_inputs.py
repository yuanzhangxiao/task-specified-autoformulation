#!/usr/bin/env python3
"""Package six preselected historical starts; do not fit or read interventions."""

import argparse
import copy
import json
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import dalla_rescue as rescue
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def build(shortlist: Path, t1_plan: Path, v7: Path, output: Path) -> dict:
    """Keep exact requests/vectors and audit the explicitly corrected saved patch."""
    catalog = sealed_read(shortlist / "selection.json")
    old = sealed_read(t1_plan)
    rows, cells, models = [], {}, {}
    for item in catalog["models"]:
        value = sealed_read(shortlist / "models" / f"{item['label']}.json")
        if value["artifact_sha256"] != item["artifact_sha256"]:
            raise ValueError("shortlisted model identity differs")
        request = PublicFitRequest.model_validate(value["request"])
        model, _, _ = public._lower(request)
        if (
            public.content_sha256(model.validated.candidate.model_dump(mode="json"))
            != value["lowered_candidate_sha256"]
        ):
            raise ValueError("shortlisted lowered candidate differs")
        models[item["label"]] = value
        cells[value["cell"]] = old["cells"][value["cell"]]
    required = {"brief_canonical_r9", "full_perturbed_r4", "full_perturbed_r13"}
    if set(models) != required:
        raise ValueError("requires the three reviewed historical candidates")
    for label in ("brief_canonical_r9", "full_perturbed_r4", "full_perturbed_r13"):
        value = models[label]
        rows.append(
            {
                "task": {
                    "task_id": label,
                    "cell": value["cell"],
                    "seed": 1,
                    "arm": "brief_only" if label.startswith("brief") else "full",
                },
                "request": value["request"],
                "parameters": value["parameters"],
                "provenance": {"source": value, "seed_kind": "saved_fitted_vector"},
            }
        )
    r4, r13 = models["full_perturbed_r4"], models["full_perturbed_r13"]
    seed = sibling_fit.compatible_seed(
        PublicFitRequest.model_validate(r13["request"]),
        PublicFitRequest.model_validate(r4["request"]),
        r13["parameters"],
        PublicSplit.model_validate(cells[r4["cell"]]["training"]),
        allow_initialization_changes=True,
    )
    if (
        seed["fresh_parameters"]
        or seed["changed_declarations"]
        or seed["changed_boundaries"]
    ):
        raise ValueError("R13 must provide an exactly compatible complete R4 start")
    transferred = copy.deepcopy(rows[1])
    transferred["task"]["task_id"] = "full_perturbed_r4_from_r13"
    transferred["parameters"] = seed["parameters"]
    transferred["provenance"] = {
        "seed_kind": "compatible_r13_transfer",
        "seed_audit": seed,
        "parent_model_sha256": r13["artifact_sha256"],
        "child_model_sha256": r4["artifact_sha256"],
    }
    rows.insert(2, transferred)
    source = sealed_read(v7 / "plan.json")
    prior = sealed_read(v7 / "results/cell00_seed0_full/round_00/result.json")
    proposal = sealed_read(v7 / "results/cell00_seed0_full/round_01/proposal.json")
    if proposal["parent_sha256"] != prior["artifact_sha256"]:
        raise ValueError("saved delay patch has a different parent")
    parent = prior["selected"]
    task = prior["task"]
    cell = source["cells"][task["cell"]]
    cells[task["cell"]] = cell
    request = PublicFitRequest.model_validate(parent["request"])
    model, _, _ = public._lower(request)
    if (
        public.content_sha256(model.validated.candidate.model_dump(mode="json"))
        != parent["fit"]["lowered_candidate_sha256"]
        or public.content_sha256(parent["request"]) != parent["fit"]["request_sha256"]
    ):
        raise ValueError("saved hard parent request/fit binding differs")
    rows.append(
        {
            "task": {**task, "task_id": "hard_parent_r14"},
            "request": parent["request"],
            "parameters": parent["fit"]["parameters"],
            "provenance": {
                "source_result_sha256": prior["artifact_sha256"],
                "seed_kind": "saved_fitted_vector",
            },
        }
    )
    correction = rescue.correct_declaration(
        parent["bundle"], proposal["attempts"][0]["raw"], "par_004", "tau_meal_rescue"
    )
    revised = correction["revision"]["bundle"]
    child = pipeline.request_for(revised, source, task, 1)
    transferred = sibling_fit.compatible_seed(
        request,
        child,
        parent["fit"]["parameters"],
        PublicSplit.model_validate(cell["training"]),
        allow_initialization_changes=True,
    )
    rows.append(
        {
            "task": {**task, "task_id": "hard_delay_corrected"},
            "request": child.model_dump(mode="json"),
            "parameters": transferred["parameters"],
            "provenance": {
                "source_proposal_sha256": proposal["artifact_sha256"],
                "source_parent_sha256": prior["artifact_sha256"],
                "correction": correction,
                "seed_audit": transferred,
                "seed_kind": "corrected_patch_with_compatible_start",
            },
        }
    )
    # Only fields used for public checks and replay enter the portable experiment.
    fields = (
        "brief",
        "context",
        "mechanism_spec",
        "target_contract",
        "training",
        "validation",
    )
    cells = {name: {k: cell[k] for k in fields} for name, cell in cells.items()}
    return sealed_write(
        output,
        {
            "protocol": rescue.INPUT_PROTOCOL,
            "rows": rows,
            "cells": cells,
            "source_shortlist_sha256": catalog["artifact_sha256"],
            "source_t1_plan_sha256": old["artifact_sha256"],
            "source_v7_plan_sha256": source["artifact_sha256"],
            "test_data_opened": False,
            "live_llm_calls": 0,
            "intervention_data_used": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--t1-plan", type=Path, required=True)
    parser.add_argument("--v7", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.shortlist, args.t1_plan, args.v7, args.output)
    print(
        json.dumps(
            {"input_sha256": value["artifact_sha256"], "tasks": len(value["rows"])},
            indent=2,
        )
    )
