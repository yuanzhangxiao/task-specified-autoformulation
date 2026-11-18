"""Revalidate the unchanged scientific critic on an isolated local Delta service."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.judging import HybridScoringConfig
from autoformalism.llm import jetstream, local_judge
from autoformalism.llm.exceptions import LLMError
from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal import judge_jetstream_calibration as original
from autoformalism.rebuttal import judge_jetstream_calibration_report as reporting
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write

PROTOCOL = "local-known-case-revalidation-1"


def plan_for(bundle: dict, image_sha256: str) -> dict:
    """Freeze all 140 original orientations and the pinned local serving assets."""
    if len(image_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in image_sha256
    ):
        raise ValueError("a serving image SHA-256 is required")
    value = original._plan(bundle)
    return value | {
        "protocol": PROTOCOL,
        "transport_policy": local_judge.POLICY,
        "endpoint": "allocation-local service, bound by runtime receipt",
        "served_model_revision": local_judge.REVISION,
        "serving_image_sha256": image_sha256,
        "settings_forwarding": (
            "pinned local vLLM; operational identity checked by launcher"
        ),
    }


def freeze(source: Path, root: Path, image_sha256: str) -> dict:
    """Create a separate calibration; historical hosted records stay unchanged."""
    if source.resolve().is_relative_to(root.resolve()):
        raise ValueError("calibration source must be outside output root")
    bundle = sealed_read(source)
    with public._lock(root):
        sealed_write(
            root / "inputs.json",
            {k: v for k, v in bundle.items() if k != "artifact_sha256"},
        )
        return sealed_write(root / "plan.json", plan_for(bundle, image_sha256))


def verify(root: Path) -> tuple[dict, dict]:
    """Reject code, cases, protocol or serving-identity drift on resume."""
    plan, bundle = sealed_read(root / "plan.json"), sealed_read(root / "inputs.json")
    if {k: v for k, v in plan.items() if k != "artifact_sha256"} != plan_for(
        bundle, plan["serving_image_sha256"]
    ):
        raise ValueError("local calibration identity changed")
    return plan, bundle


def run(root: Path, base_url: str, key_supplier, progress=print) -> None:
    """Resume only unstarted orientations, preserving physical request budgets."""
    local_judge.checked_url(base_url)
    with public._lock(root):
        plan, bundle = verify(root)
        for number, (name, raw, repetition, order) in enumerate(
            original.units(bundle), 1
        ):
            work = root / "orientations" / name
            identity = {"plan": plan["artifact_sha256"], "unit": name}
            path = work / "result.json"
            if path.exists():
                if sealed_read(path)["identity"] != identity:
                    raise ValueError("local calibration unit identity changed")
                continue
            marker = work / "started.json"
            if marker.exists():
                if sealed_read(marker)["identity"] != identity:
                    raise ValueError("local calibration reservation changed")
                sealed_write(path, {"identity": identity, "status": "interrupted"})
                continue
            if not key_supplier().strip():
                raise ValueError("missing local service credential")
            sealed_write(marker, {"identity": identity})
            progress(f"Local critic calibration {number}/140: {name}")
            transport = local_judge.LocalJudgeTransport(
                work / "calls", base_url, key_supplier
            )
            try:
                entry = bundle["contexts"][f"{raw['benchmark_id']}/{raw['tier']}"]
                row = original._assess(
                    raw,
                    entry,
                    repetition,
                    order,
                    work,
                    transport,
                    client_factory=local_judge.client,
                )
                result = {"identity": identity, "status": "reviewed", "row": row}
            except LLMError as error:
                result = {
                    "identity": identity,
                    "status": "unavailable",
                    "error_type": type(error).__name__,
                }
            sealed_write(path, result)
            if result["status"] != "reviewed":
                break


def report(root: Path) -> dict:
    """Use the same existing gates, with all missing orientations counted."""
    plan, bundle = verify(root)
    rows, failures, statuses = [], [], Counter()
    for name, raw, repetition, order in original.units(bundle):
        path = root / "orientations" / name / "result.json"
        value = sealed_read(path) if path.exists() else {"status": "pending"}
        if path.exists() and value["identity"] != {
            "plan": plan["artifact_sha256"],
            "unit": name,
        }:
            raise ValueError("calibration result identity changed")
        statuses[value["status"]] += 1
        if value["status"] == "reviewed":
            rows.append(value["row"])
        else:
            failures.append(
                {
                    "pair_id": raw["pair_id"],
                    "judge_model": original.MODEL,
                    "repetition": repetition,
                    "order": order,
                    "status": value["status"],
                }
            )
    current = reporting._evaluation(
        rows,
        failures,
        bundle,
        HybridScoringConfig(**plan["scientific_protocol"]["scoring"]),
    )
    finished = not statuses["pending"]
    value = {
        "protocol": PROTOCOL,
        "identity": plan["artifact_sha256"],
        "status": "complete" if finished else "incomplete",
        "status_counts": dict(statuses),
        "planned_orientations": 140,
        "known_case_gate_passed": current["gate_assessment"]["passed"]
        if finished
        else None,
        "current": current,
        "cost": reporting._cost(root),
        "fresh_holdout_calibration_established": False,
        "test_data_opened": False,
    }
    atomic_json(root / "summary.json", value)
    return value


def bind(root: Path, campaign: dict, calibration_root: Path) -> dict:
    """Bind production only after the full local known-case checks have passed."""
    plan, _ = verify(calibration_root)
    summary = report(calibration_root)
    if (
        campaign.get("critic_transport") != local_judge.POLICY
        or summary["status"] != "complete"
        or summary["known_case_gate_passed"] is not True
        or campaign["serving_image_sha256"] != plan["serving_image_sha256"]
        or plan["output_contract"] != jetstream.OUTPUT_CONTRACT
    ):
        raise ValueError(
            "complete compatible local calibration with passed gates required"
        )
    return sealed_write(
        root / "critic_authorization.json",
        {
            "identity": campaign["artifact_sha256"],
            "calibration_plan_sha256": plan["artifact_sha256"],
            "summary_sha256": public.content_sha256(summary),
            "known_case_gate_passed": True,
            "fresh_calibration_claim": False,
            "transport_policy": local_judge.POLICY,
            "output_contract": jetstream.OUTPUT_CONTRACT,
            "model_revision": local_judge.REVISION,
            "serving_image_sha256": plan["serving_image_sha256"],
        },
    )
