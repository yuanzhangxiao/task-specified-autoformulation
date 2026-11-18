"""Export fitted, targeted Candidate-15/21 alternatives separately from searches."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts.collect_iclr_latent_histories import digest, read_sealed


def fitted_model(problem: dict, parameters: dict, brief: dict, aliases: dict) -> dict:
    """Exclude trajectory initializer roots from the structural coefficient map."""
    candidate = problem["base_candidate"]
    return {
        "candidate": candidate,
        "context": problem["context"],
        "parameters": {
            p["name"]: parameters[p["name"]] for p in candidate["parameters"]
        },
        "brief": brief,
        "modeled_observed_auxiliaries": [
            name
            for public, name in aliases.items()
            if public in problem["context"]["auxiliaries"]
        ],
    }


def collect(project: Path) -> dict:
    """Retain failed alternatives and independently authenticate fitted children."""
    root = project / "output/t2-breadth-results-20261108/returned"
    plan = read_sealed(root / "plan.json")
    inputs = read_sealed(root / "inputs.json")
    fit_path = project / "output/iclr-method-freeze-20261109/frozen-run/models.json"
    fits = read_sealed(fit_path)
    by_name = {m["task"]: m for m in fits["models"] if m["task"] != "c15-4-zero-basal"}
    models = {}

    def register(problem: dict, parameters: dict) -> str:
        m = fitted_model(problem, parameters, inputs["brief"], inputs["aliases"])
        uid = digest(m)
        models[uid] = m
        return uid

    parents = {
        "c15": register(inputs["c15"]["problem"], inputs["c15_training_parameters"]),
        "c21": register(
            inputs["c21"]["problem"],
            inputs["c21"]["parameters_including_training_roots"],
        ),
    }
    request_path = (
        project / "output/iclr-candidate15-equation-export/proposer-request.json"
    )
    request = json.loads(request_path.read_text())
    payload = json.loads(
        next(m["content"] for m in request["body"]["messages"] if m["role"] == "user")
    )
    runs = []
    for p in sorted((root / "results").glob("*.json")):
        result = read_sealed(p)
        name, group = result["name"], result["group"]
        visits = []
        if group in parents:
            visits.append({"round": 0, "selected_model": parents[group]})
        final = None
        if name in by_name:
            fit = by_name[name]
            if digest(result["problem"]) != digest(fit["problem"]):
                raise ValueError("guided fit differs from admitted alternative")
            if digest(fit["problem"], compact=True) != fit["model_sha256"]:
                raise ValueError("guided frozen model hash mismatch")
            # The sealed freeze authenticates shared coefficients. Its legacy
            # parameter hash also covers training roots omitted from this export.
            final = register(fit["problem"], fit["shared"])
            raw = (
                result.get("checks", {})
                .get("audit", {})
                .get("normalization", {})
                .get("raw_reply", {})
            )
            visits.append(
                {
                    "round": 1,
                    "selected_model": final,
                    "guided_feedback": {
                        "status": "targeted_training_guidance",
                        "numerical_evidence_cited": False,
                        "numerical_feedback_supplied": group == "c15",
                        "hypothesis": raw.get("hypothesis"),
                        "reply_evidence_refs": raw.get("evidence_ids", []),
                        "scientific_advice_present": True,
                        "request_sha256": digest(request) if group == "c15" else None,
                        "training_fields": sorted(payload) if group == "c15" else [],
                    },
                }
            )
        runs.append(
            {
                "task": {
                    "task_id": name,
                    "cell": "phase_c_dalla_man_t2_canonical_named_easy_rates_v1",
                    "seed": result.get("initial_sampling_seed"),
                    "treatment": group + "_guided_alternative",
                },
                "visits": visits,
                "final_status": "complete" if final else result["status"],
                "generation_status": result["status"],
                "result_sha256": result["artifact_sha256"],
                "search_complete": True,
                "final_model": final,
                "latest_model": final,
            }
        )
    return {
        "protocol": "iclr-guided-latent-history-1",
        "root": str(root),
        "plan_sha256": plan["artifact_sha256"],
        "model_settings": plan["settings"],
        "rounds": 2,
        "runs": runs,
        "models": models,
        "errors": [],
        "frozen_sha256": fits["artifact_sha256"],
        "guided_request": payload,
        "note": (
            "Alternatives share fitted parents; not independent search seeds. "
            "The zero-basal deterministic control is excluded."
        ),
        "test_data_opened": False,
        "llm_calls": 0,
        "model_fits": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = collect(args.project)
    result["artifact_sha256"] = digest(result)
    args.output.write_bytes(
        gzip.compress(json.dumps(result, sort_keys=True).encode(), mtime=0)
    )


if __name__ == "__main__":
    main()
