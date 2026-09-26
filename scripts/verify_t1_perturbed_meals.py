#!/usr/bin/env python3
"""Cross-check frozen meal rollouts and flag solver-sensitive paired controls."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.fitting.simulation import simulate_trajectory
from scripts import compare_t1_perturbed_meals as probe


def metric_reliability(case: dict, agrees: dict[str, bool]) -> dict[str, bool | None]:
    """A paired metric needs numerical agreement for both of its trajectories."""
    own = agrees.get(case["id"], False)
    return {
        "absolute_verified": own,
        "response_verified": own and agrees.get(case["control"], False)
        if case["control"]
        else None,
        "schedule_difference_verified": own and agrees.get(case["comparison"], False)
        if case["comparison"]
        else None,
    }


def run(root: Path) -> dict:
    """Persist an independent DOP853 replay; do not revise frozen main results."""
    plan = probe.sealed_read(root / "plan.json")
    summary = probe.sealed_read(root / "summary.json")
    if summary["plan_sha256"] != plan["artifact_sha256"]:
        raise ValueError("summary belongs to another plan")
    if plan["identity"] != probe.identity():
        raise ValueError("original frozen runtime changed")
    config = probe.SETTINGS.model_copy(
        update={
            "integration_method": "DOP853",
            "relative_tolerance": 1e-9,
            "absolute_tolerance": 1e-11,
        }
    )
    refs = {
        c["id"]: probe.sealed_read(root / "references" / f"{c['id']}.json")["row"]
        for c in plan["cases"]
    }
    refs["conditional_single60"] = probe.conditional_row(
        refs["single_60"], refs["fasting_300"]
    )
    checks, rows = [], []
    for job in plan["models"]:
        model = probe.compile_candidate(
            probe.CandidateModel.model_validate(job["candidate"]),
            probe.ValidationContext.model_validate(job["context"]),
        )
        agrees = {}
        for key, row in refs.items():
            old_path = (
                root / "conditional" / f"{job['id']}.json"
                if key == "conditional_single60"
                else root / "replays" / job["id"] / f"{key}.json"
            )
            old = probe.sealed_read(old_path)
            path = root / "verification-replays" / job["id"] / f"{key}.json"
            if path.exists():
                new = probe.sealed_read(path)
                if new["plan_sha256"] != plan["artifact_sha256"]:
                    raise ValueError("verification checkpoint belongs to another plan")
                if new["settings"] != config.model_dump(mode="json"):
                    raise ValueError("verification solver settings changed")
            else:
                sim = simulate_trajectory(
                    model,
                    probe.trajectory(row),
                    job["parameters"],
                    job["initials"],
                    config,
                    deadline=monotonic() + 120,
                    reset_observed_states=False,
                )
                pred = sim.predictions.get("v01")
                success = bool(
                    sim.success and pred is not None and np.isfinite(pred).all()
                )
                new = probe.sealed_write(
                    path,
                    {
                        "plan_sha256": plan["artifact_sha256"],
                        "settings": config.model_dump(mode="json"),
                        "success": success,
                        "message": sim.message,
                        "predicted": pred.tolist() if success else None,
                    },
                )
            success = new["success"] and old["success"]
            a, b = np.asarray(new["predicted"]), np.asarray(old["predicted"])
            agrees[key] = bool(success and np.allclose(a, b, rtol=1e-5, atol=1e-4))
            checks.append(
                {
                    "model_id": job["id"],
                    "case_id": key,
                    "agrees": agrees[key],
                    "max_abs_difference": float(np.max(np.abs(a - b)))
                    if success
                    else None,
                }
            )
        for case in plan["cases"]:
            row = next(
                r
                for r in summary["rows"]
                if r["model_id"] == job["id"] and r["case_id"] == case["id"]
            )
            flags = metric_reliability(case, agrees)
            rows.append(
                {
                    "model_id": job["id"],
                    "case_id": case["id"],
                    **flags,
                    "verified_nmse": row["nmse"]
                    if flags["absolute_verified"]
                    else None,
                    "verified_response_error": (row["response"] or {}).get(
                        "relative_squared_error"
                    )
                    if flags["response_verified"]
                    else None,
                    "verified_schedule_difference_error": (
                        row["schedule_difference"] or {}
                    ).get("relative_squared_error")
                    if flags["schedule_difference_verified"]
                    else None,
                }
            )
        print(
            job["id"], "verified", sum(agrees.values()), "of", len(agrees), flush=True
        )
    probe.write_csv(root / "verified-metrics.csv", rows)
    return probe.sealed_write(
        root / "solver-audit.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "settings": config.model_dump(mode="json"),
            "agreement_rtol": 1e-5,
            "agreement_atol": 1e-4,
            "checks": checks,
            "verified_metric_rows": rows,
            "agreed": sum(c["agrees"] for c in checks),
            "total": len(checks),
            "verification_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "limitation": (
                "Paired scores are withheld if either trajectory is solver-sensitive. "
                "Original primary results remain intact."
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(run(parser.parse_args().root)["artifact_sha256"])
