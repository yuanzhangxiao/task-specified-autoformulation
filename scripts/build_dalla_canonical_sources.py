#!/usr/bin/env python3
"""Export two selected historical starts and their public development data."""

import argparse
import copy
import json
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import dalla_canonical_rescue as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.staged_topology import content_hash

SELECTED = (
    (
        "brief_canonical_r9",
        "4d630f90e574895a0046686a7e0f1380bf99cb1dc154e399bc9ff33f360d660d",
        "brief_only",
        1,
    ),
    (
        "full_canonical_r2",
        "076b95891e38358230e28131f2aad822bab81e74c18cdbf925b59d495e27ee98",
        "full",
        0,
    ),
)


def build(inventory: Path, public_plan: Path, output: Path) -> dict:
    """Verify saved identities without importing scores or reference data."""
    catalog, plan = sealed_read(inventory), sealed_read(public_plan)
    if (
        catalog.get("test_files_opened") is not False
        or plan.get("test_data_opened") is not False
    ):
        raise ValueError("requires public development inventory and data plan")
    rows, cells = [], {}
    for label, ident, arm, seed in SELECTED:
        source = catalog["models"][ident]
        request = PublicFitRequest.model_validate(source["request"])
        model, _, _ = public._lower(request)
        lowered = public.content_sha256(
            model.validated.candidate.model_dump(mode="json")
        )
        if lowered != source["lowered_candidate_sha256"]:
            raise ValueError("saved lowered candidate differs")
        cell = plan["cells"][source["cell"]]
        if (
            cell["brief"] != source["public_brief"]
            or cell["target_contract"] != source["target_contract"]
        ):
            raise ValueError("public task contract differs")
        fields = (
            "brief",
            "context",
            "mechanism_spec",
            "target_contract",
            "training",
            "validation",
        )
        cells[source["cell"]] = {k: copy.deepcopy(cell[k]) for k in fields}
        sibling_fit.compatible_seed(
            request,
            request,
            source["parameters"],
            PublicSplit.model_validate(cell["training"]),
        )
        rows.append(
            {
                "task": {
                    "task_id": label,
                    "cell": source["cell"],
                    "arm": arm,
                    "seed": seed,
                },
                "request": request.model_dump(mode="json"),
                "parameters": source["parameters"],
                "provenance": {
                    "inventory_sha256": catalog["artifact_sha256"],
                    "model_id": ident,
                    "source_model_sha256": content_hash(source),
                    "lowered_candidate_sha256": lowered,
                },
            }
        )
    return sealed_write(
        output,
        {
            "protocol": campaign.SOURCE_PROTOCOL,
            "rows": rows,
            "cells": cells,
            "test_data_opened": False,
            "intervention_data_used": False,
            "source_public_plan_sha256": plan["artifact_sha256"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--public-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.inventory, args.public_plan, args.output)
    print(
        json.dumps(
            {"source_sha256": result["artifact_sha256"], "models": len(result["rows"])}
        )
    )
