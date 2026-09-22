#!/usr/bin/env python3
"""Export saved Full/Brief-only Dalla Man fits without data, fitting or LLM calls.

Uses only the Python standard library. Supports review-deadline-1 through -5;
other campaign formats are reported explicitly, never treated as empty runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

PROTOCOLS = {f"review-deadline-{i}" for i in range(1, 6)}
ARMS = {"full", "brief_only"}
CELL = re.compile(
    r"phase_b_(?:dalla_man_t[1-4]_(?:canonical|perturbed)_named|"
    r"anonymous_system_t[1-4]_(?:canonical|perturbed)_obfuscated)_(?:easy|hard)"
)
FIT_FIELDS = (
    "protocol",
    "profile",
    "status",
    "message",
    "parameters",
    "lowered_candidate_sha256",
    "initialization_plan_sha256",
    "request_sha256",
    "identity",
    "training",
    "validation",
    "budget_exhausted",
    "native_optimizer_converged",
    "actual_residual_calls",
    "independent_replay",
    "training_only_parameter_estimation",
    "validation_initials_fitted",
)
REQUEST_FIELDS = (
    "protocol",
    "profile",
    "base_candidate",
    "context",
    "initialization_plan",
    "parameter_guesses",
    "random_seed",
    "source",
)


def digest(value: object) -> str:
    """Match the historical content_hash convention exactly."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_sealed(path: Path) -> dict:
    """Refuse incomplete, modified or unsealed historical artifacts."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("artifact_sha256") != digest(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    ):
        raise ValueError(f"artifact digest differs: {path}")
    return value


def fit_payload(endpoint: dict) -> dict:
    """Copy equations and saved fit evidence, excluding residual packets/test scores."""
    request, fit = endpoint["request"], endpoint["fit"]
    required = {"base_candidate", "context", "initialization_plan"}
    if not required <= request.keys():
        raise ValueError("incomplete fit request")
    parameters = fit.get("parameters")
    if not isinstance(parameters, dict) or not fit.get("lowered_candidate_sha256"):
        raise ValueError("missing fitted parameter vector or candidate identity")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in parameters.values()
    ):
        raise ValueError("nonfinite or nonnumeric fitted parameter")
    return {
        "request": {k: request[k] for k in REQUEST_FIELDS if k in request},
        "fit": {k: fit[k] for k in FIT_FIELDS if k in fit},
        "certificate": endpoint.get("certificate"),
        "origin_round": endpoint.get("origin_round"),
        "origin_task": endpoint.get("origin_task"),
        "fit_result_sha256": endpoint.get("fit_result_sha256"),
    }


def collect(roots: list[Path]) -> dict:
    """Keep every saved round and both retained/trial roles; never select a winner."""
    campaigns, models, occurrences, errors = [], {}, [], []
    for root in sorted({p.expanduser().resolve() for p in roots}):
        campaign = {"root": str(root)}
        campaigns.append(campaign)
        try:
            # Inspect only plan.json, never summary/test/evaluation files.
            raw = json.loads((root / "plan.json").read_text())
            if not isinstance(raw, dict):
                raise ValueError("plan must be a JSON object")
            campaign["protocol"] = raw.get("protocol")
            if raw.get("protocol") not in PROTOCOLS:
                campaign["status"] = "unsupported_protocol"
                campaign["declared_dalla_tasks"] = [
                    {k: t.get(k) for k in ("task_id", "cell", "arm", "seed")}
                    for t in raw.get("tasks", [])
                    if isinstance(t, dict) and CELL.fullmatch(str(t.get("cell", "")))
                ]
                continue
            plan = read_sealed(root / "plan.json")
            tasks = [
                t
                for t in plan["tasks"]
                if t.get("arm") in ARMS and CELL.fullmatch(t.get("cell", ""))
            ]
            campaign.update(
                status="inventoried",
                plan_sha256=plan["artifact_sha256"],
                tasks=len(tasks),
                planned_rounds=plan["config"]["rounds"],
                source_round=plan.get("continuation", {}).get("source_round", 0),
            )
            for task in tasks:
                task_id = task["task_id"]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
                    raise ValueError("unsafe task identifier")
                cell_info = plan["cells"][task["cell"]]
                for index in range(campaign["planned_rounds"]):
                    path = root / "results" / task_id / f"round_{index:02d}/result.json"
                    row = {
                        "campaign": str(root),
                        "plan_sha256": plan["artifact_sha256"],
                        "task": task,
                        "phase_round": index,
                        "global_round": index + campaign["source_round"],
                        "source_path": str(path),
                        "selected": None,
                        "trial": None,
                    }
                    occurrences.append(row)
                    if not path.exists():
                        row["status"] = "checkpoint_missing"
                        continue
                    try:
                        result = read_sealed(path)
                        if result["task"] != task or result["round"] != index:
                            raise ValueError("checkpoint task or round differs")
                        row.update(
                            status=result["status"],
                            source_result_sha256=result["artifact_sha256"],
                        )
                        for role in ("selected", "trial"):
                            endpoint = result.get(role)
                            if endpoint is None:
                                continue
                            payload = fit_payload(endpoint)
                            identity = digest(
                                {
                                    "cell": task["cell"],
                                    "request": payload["request"],
                                    "parameters": payload["fit"]["parameters"],
                                    "lowered_candidate_sha256": payload["fit"][
                                        "lowered_candidate_sha256"
                                    ],
                                }
                            )
                            models.setdefault(
                                identity,
                                {
                                    "cell": task["cell"],
                                    "request": payload["request"],
                                    "parameters": payload["fit"]["parameters"],
                                    "lowered_candidate_sha256": payload["fit"][
                                        "lowered_candidate_sha256"
                                    ],
                                    "public_brief": cell_info.get("brief"),
                                    "target_contract": cell_info.get("target_contract"),
                                },
                            )
                            # Keep per-fit metrics/provenance even when vectors repeat.
                            row[role] = {
                                "model_id": identity,
                                **{k: v for k, v in payload.items() if k != "request"},
                            }
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        row["status"] = "export_error"
                        row["error"] = str(exc)
                        errors.append({"path": str(path), "error": str(exc)})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            campaign.update(status="inventory_error", error=str(exc))
            errors.append({"path": str(root), "error": str(exc)})

    coverage = []
    for task in range(1, 5):
        for dynamics in ("canonical", "perturbed"):
            for family, variant in (
                ("dalla_man", "named"),
                ("anonymous_system", "obfuscated"),
            ):
                for tier in ("easy", "hard"):
                    cell = f"phase_b_{family}_t{task}_{dynamics}_{variant}_{tier}"
                    for arm in sorted(ARMS):
                        rows = [
                            r
                            for r in occurrences
                            if r["task"]["cell"] == cell and r["task"]["arm"] == arm
                        ]
                        coverage.append(
                            {
                                "cell": cell,
                                "arm": arm,
                                "planned_lineages": len(
                                    {
                                        (r["campaign"], r["task"]["task_id"])
                                        for r in rows
                                    }
                                ),
                                "unique_retained_fits": len(
                                    {
                                        r["selected"]["model_id"]
                                        for r in rows
                                        if r["selected"]
                                    }
                                ),
                                "unique_trial_fits": len(
                                    {r["trial"]["model_id"] for r in rows if r["trial"]}
                                ),
                                "missing_checkpoints": sum(
                                    r["status"] == "checkpoint_missing" for r in rows
                                ),
                            }
                        )
    value = {
        "protocol": "dalla-fitted-model-inventory-1",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "partial" if errors else "complete_for_supported_roots",
        "campaigns": campaigns,
        "coverage": coverage,
        "models": models,
        "occurrences": occurrences,
        "errors": errors,
        "live_llm_calls": 0,
        "parameter_fitting_performed": False,
        "trajectory_files_opened": False,
        "test_files_opened": False,
        "limitation": (
            "Saved vectors, not a scientific verdict or new model selection. Only "
            "review-deadline-1 through -5 are supported; absent coverage means no "
            "exported model in the scanned roots, not proof no run exists elsewhere. "
            "All retained/trial rounds stay labeled. Compilation and lowered-model "
            "identity must be checked before using any exported equation."
        ),
    }
    return {**value, "artifact_sha256": digest(value)}


def main() -> None:
    """Write an immutable compact JSON bundle suitable for downloading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--scan", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = list(args.root)
    for directory in args.scan:
        if not directory.is_dir():
            parser.error(f"scan directory missing: {directory}")
        roots.extend(
            p for p in directory.iterdir() if p.is_dir() and (p / "plan.json").is_file()
        )
    if not roots:
        parser.error("no campaign roots found")
    value = collect(roots)
    if args.out.exists() and json.loads(args.out.read_text()) != value:
        parser.error("output differs; use a new --out path for a later snapshot")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "status": value["status"],
                "saved_request_and_parameter_records": len(value["models"]),
                "campaign_statuses": dict(
                    Counter(c["status"] for c in value["campaigns"])
                ),
                "covered_cells": sorted({m["cell"] for m in value["models"].values()}),
                "errors": value["errors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
