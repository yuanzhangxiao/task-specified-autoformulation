"""Read-only, standard-library collector for retained latent-state histories.

Run on the existing campaign filesystem; stdout is one gzip-compressed JSON.
No campaign code, model execution, provider call, fitting, or test-data access.
"""

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path


def digest(value: object, *, compact: bool = False) -> str:
    """Match campaign seals, or the public fitter's compact identities."""
    options = {"separators": (",", ":")} if compact else {}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, **options).encode()
    ).hexdigest()


def read_sealed(path: Path) -> dict:
    """Reject modified or partially written scientific receipts."""
    d = json.loads(path.read_text())
    if d.get("artifact_sha256") != digest(
        {k: v for k, v in d.items() if k != "artifact_sha256"}
    ):
        raise ValueError(f"invalid campaign seal: {path}")
    return d


def model_record(endpoint: dict) -> dict:
    """Preserve the exact fitted model and public specification, not split arrays."""
    bundle, fit = endpoint["bundle"], endpoint["fit"]
    candidate = bundle["candidate"]
    if fit.get("lowered_candidate_sha256") not in (
        None,
        digest(candidate, compact=True),
    ):
        raise ValueError("fitted candidate identity mismatch")
    initial = bundle.get("initialization") or {}
    return {
        "candidate": candidate,
        "context": initial.get("context") or bundle["context"],
        "parameters": fit["parameters"],
        "brief": bundle.get("brief"),
        "initialization_plan": (endpoint.get("request") or {}).get(
            "initialization_plan", initial.get("plan")
        ),
        "fit_status": fit["status"],
        "training": fit.get("training"),
        "validation": fit.get("validation"),
    }


def request_projection(path: Path, expected: str) -> dict:
    """Read only scientific request-body fields; omit credentials and headers."""
    if not path.exists():
        return {"status": "missing", "source": str(path)}
    raw = path.read_bytes()
    call = json.loads(raw)
    if call.get("request_hash") != expected:
        raise ValueError("proposer call identity mismatch")
    request = call["request"]
    messages = request.get("body", request).get("messages", [])
    payloads = []
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(message["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    if not payloads:
        return {"status": "unparsed", "source": str(path)}
    payload = payloads[-1]
    fields = (
        "protocol",
        "stage",
        "evidence_catalog",
        "training_evidence",
        "scientific_advice",
        "public_requirement_findings",
        "retry_feedback",
        "current_model_size",
        "state_capabilities",
        "initialization_plan",
    )
    return {
        "status": "available",
        "source": str(path),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "request_hash": expected,
        "payload": {k: payload[k] for k in fields if k in payload},
    }


def pruning_model(directory: Path, selected: str, fit: dict, parent: dict) -> dict:
    """Recover the chosen lowered model from its saved fit freeze if necessary."""
    if selected == "parent":
        return {
            **parent,
            "parameters": fit["parameters"],
            "training": fit.get("training"),
            "validation": fit.get("validation"),
        }
    path = directory / selected / "freeze.json"
    outer = json.loads(path.read_text())
    for d in [outer, outer.get("public_fit", outer)]:
        if d.get("identity") != digest(
            {k: v for k, v in d.items() if k != "identity"}, compact=True
        ):
            raise ValueError("invalid pruning fit-freeze identity")
    frozen = outer.get("public_fit", outer)
    candidate = frozen["lowered_candidate"]
    if digest(candidate, compact=True) != fit["lowered_candidate_sha256"]:
        raise ValueError("pruning candidate differs from selected fit")
    return {
        **parent,
        "candidate": candidate,
        "context": frozen["lowered_context"],
        "parameters": fit["parameters"],
        "fit_status": fit["status"],
        "initialization_plan": frozen["request"].get("initialization_plan"),
        "training": fit.get("training"),
        "validation": fit.get("validation"),
    }


def collect(root: Path) -> dict:
    """Collect each planned run, its selected lineage and finalization status."""
    plan = read_sealed(root / "plan.json")
    old = plan["protocol"] == "final-component-campaign-1"
    if not old and plan["protocol"] != "iclr-controlled-search-1":
        raise ValueError("unsupported campaign protocol")
    tasks = plan["tasks"] if old else [t for b in plan["blocks"] for t in b["tasks"]]
    rounds = plan["config"]["rounds"] if old else plan["visits"]
    config = plan["config"] if old else plan
    models, runs, errors = {}, [], []
    frozen_path = root / "frozen_models.json"
    frozen = read_sealed(frozen_path) if frozen_path.exists() else None
    frozen_rows = {r["source_id"]: r for r in (frozen or {}).get("rows", [])}
    if frozen and frozen["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("frozen endpoint belongs to a different plan")

    def register(record: dict) -> str:
        uid = digest(record)
        models.setdefault(uid, record)
        return uid

    for task in tasks:
        tid = task["task_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tid):
            raise ValueError("unsafe task identifier")
        run = {"task": task, "visits": [], "final_status": "not_started"}
        previous = None
        for i in range(rounds):
            directory = root / "results" / tid / f"round_{i:02d}"
            path = directory / "result.json"
            if not path.exists():
                continue
            try:
                result = read_sealed(path)
                if result["task"]["task_id"] != tid or result["round"] != i:
                    raise ValueError("task/visit binding mismatch")
                selected = result.get("selected")
                mid = register(model_record(selected)) if selected else None
                before = (
                    set()
                    if previous is None
                    else {s["name"] for s in models[previous]["candidate"]["states"]}
                )
                after = (
                    set()
                    if mid is None
                    else {s["name"] for s in models[mid]["candidate"]["states"]}
                )
                visit = {
                    "round": i,
                    "result_sha256": result["artifact_sha256"],
                    "status": result["status"],
                    "selected_model": mid,
                    "selected_new_trial": result.get("selected_new_trial"),
                    "origin_round": (selected or {}).get("origin_round"),
                    "origin_task": (selected or {}).get("origin_task"),
                    "states_added": sorted(after - before),
                    "states_removed": sorted(before - after),
                }
                pp = directory / "proposal.json"
                if pp.exists():
                    proposal = read_sealed(pp)
                    visit["proposal_sha256"] = proposal["artifact_sha256"]
                    visit["proposal_status"] = proposal.get("status")
                    visit["attempts"] = proposal.get("attempts", [])
                    # All successful additions need request-level attribution.
                    if i > 0 and after - before and mid != previous:
                        visit["requests"] = []
                        for attempt in proposal.get("attempts", []):
                            h = attempt.get("request_hash")
                            if isinstance(h, str) and re.fullmatch(r"[a-f0-9]{64}", h):
                                visit["requests"].append(
                                    request_projection(
                                        directory / "calls" / f"{h}.json", h
                                    )
                                )
                run["visits"].append(visit)
                previous = mid
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append({"task": tid, "round": i, "error": str(exc)})
        run["latest_model"] = previous
        run["search_complete"] = len(run["visits"]) == rounds
        run["final_model"] = None
        if run["visits"]:
            run["final_status"] = (
                "pruning_pending" if run["search_complete"] else "search_incomplete"
            )
        pp = root / "pruning" / tid / "result.json"
        if run["search_complete"] and pp.exists():
            try:
                pruning = read_sealed(pp)
                if pruning["task"]["task_id"] != tid:
                    raise ValueError("pruning task differs")
                run["pruning_sha256"] = pruning["artifact_sha256"]
                selection = pruning.get("selection")
                run["pruning_selection"] = (selection or {}).get("selected")
                if selection and previous:
                    final = pruning_model(
                        pp.parent,
                        selection["selected"],
                        selection["fit"],
                        models[previous],
                    )
                    run["final_model"] = register(final)
                    run["final_status"] = "complete"
                    if task.get("critic") and not (pp.parent / "critic.json").exists():
                        run["final_status"] = "final_review_pending"
                elif not previous:
                    run["final_status"] = "no_retained_model"
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append({"task": tid, "stage": "pruning", "error": str(exc)})
        elif run["search_complete"] and not config.get("pruning_enabled", old):
            run["final_model"] = previous
            run["final_status"] = "complete" if previous else "no_retained_model"
        # Frozen exports are authoritative and survive solver-directory archiving.
        if tid in frozen_rows:
            fr = frozen_rows[tid]
            run["frozen_status"] = fr["status"]
            if fr["status"] == "ready":
                prior = models[previous] if previous else {}
                final = {
                    **prior,
                    "candidate": fr["candidate"],
                    "context": fr["context"],
                    "parameters": fr["parameters"],
                    "fit_status": "complete",
                    "training": {"normalized_mse": fr.get("training_nmse")},
                    "validation": {"normalized_mse": fr.get("validation_nmse")},
                }
                run["final_model"] = register(final)
                run["final_status"] = "complete"
                run["frozen_model_sha256"] = fr["model_sha256"]
                if digest(fr["candidate"], compact=True) != fr["model_sha256"]:
                    raise ValueError("frozen model identity mismatch")
            elif fr["status"] == "no_retained_model":
                run["final_status"] = "no_retained_model"
        runs.append(run)
    return {
        "protocol": "iclr-all-latent-history-1",
        "root": str(root),
        "plan_sha256": plan["artifact_sha256"],
        "campaign_protocol": plan["protocol"],
        "model_settings": config.get("model_settings"),
        "rounds": rounds,
        "source_sha256": plan.get("source_sha256"),
        "frozen_sha256": (frozen or {}).get("artifact_sha256"),
        "runs": runs,
        "models": models,
        "errors": errors,
        "test_data_opened": False,
        "llm_calls": 0,
        "model_fits": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    value = collect(args.root)
    value["artifact_sha256"] = digest(value)
    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", mtime=0) as stream:
        stream.write(json.dumps(value, sort_keys=True).encode())


if __name__ == "__main__":
    main()
