"""Apply the common public functional rubric to the final corrected reactor cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_campaign, mechanism_functional_campaign
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.iclr_corrected_cstr_test import intake
from scripts.iclr_cstr_correction import CSTR, EXPECTED
from scripts.iclr_submission_evidence import REPO, digest, table


def score_row(entry, subject):
    """Adapt every frozen subject while retaining unavailable endpoints."""
    t = entry["task"]
    row = {
        "benchmark_id": CSTR,
        "tier": "easy",
        "repetition": t["seed"],
        "method": component_campaign.method_name(t, "final"),
        "source_id": entry["subject_id"],
        "source_sha256": entry["source_sha256"],
        "public_prompt_sha256": EXPECTED["proposer_prompt.txt"],
        "source_public_prompt_verified": True,
        "semantics": "continuous_time",
        "semantics_basis": "final frozen ODE",
    }
    if not entry["model_available"]:
        if subject is not None:
            raise ValueError("unexpected model for unavailable final row")
        return row | {
            "status": "unavailable",
            "source_terminal_status": "unavailable",
            "error": "final fitted subject unavailable; detailed cause not in freeze",
        }
    if (
        subject is None
        or subject.subject_id != entry["subject_id"]
        or subject.method != row["method"]
        or subject.benchmark_id != CSTR
        or subject.repetition != row["repetition"]
        or subject.source_provenance.source_sha256 != entry["source_sha256"]
    ):
        raise ValueError("subject belongs to another final endpoint")
    candidate = subject.candidate
    if (
        hashlib.sha256(candidate.model_dump_json().encode()).hexdigest()
        != subject.source_provenance.candidate_sha256
    ):
        raise ValueError("candidate differs from the frozen source hash")
    return row | {
        "status": "ready",
        "source_terminal_status": "ready",
        "candidate": candidate.model_dump(mode="json"),
        "context": subject.validation_context.model_dump(mode="json"),
        "parameters": subject.parameterization.global_parameters,
        "initials": subject.parameterization.global_initial_conditions,
    }


def run(root: Path, staging: Path, out: Path) -> dict:
    """Use corrected training trajectories for the common post-freeze rubric."""
    if any(
        out.resolve().is_relative_to(p.resolve())
        or p.resolve().is_relative_to(out.resolve())
        for p in (root, staging)
    ):
        raise ValueError(
            "use a dedicated output outside the campaign and development roots"
        )
    campaign, frozen, subjects, correction = intake(root, staging)
    cfg_path = REPO / "configs/mechanism_functional_v1.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["cells"] = {CSTR: cfg["cells"][CSTR]}
    rows = [score_row(e, subjects.get(e["subject_id"])) for e in frozen["roster"]]
    folder = root / "public" / "phase_b_v1" / CSTR
    if any(digest(folder / name) != sha for name, sha in EXPECTED.items()):
        raise ValueError("corrected reactor development files changed")
    with public._lock(out):
        assessed = mechanism_functional_campaign.prepare(
            rows,
            cfg,
            out,
            root / "public",
            {
                "selection": frozen["artifact_sha256"],
                "plan": campaign["artifact_sha256"],
                "correction": correction,
                "rubric_sha256": digest(cfg_path),
                "adapter_sha256": digest(Path(__file__)),
            },
        )
        for i in range(len(rows)):
            mechanism_functional_campaign.run(out, i)
        result = mechanism_functional_campaign.report(out)
        details = []
        for source, r in zip(rows, result["rows"], strict=True):
            rr = sealed_read(out / "results" / f"{r['index']:04d}" / "result.json")
            for finding in r["mechanisms"]:
                details.append(
                    {
                        "source_id": source["source_id"],
                        "endpoint": source["source_sha256"],
                        "method": r["method"],
                        "seed": r["repetition"],
                        "availability": source["status"],
                        "mechanism": finding["id"],
                        "status": finding["status"],
                        "reason": finding.get("reason"),
                        "receipt": rr["artifact_sha256"],
                    }
                )
        table(out / "corrected_cstr_compliance.csv", details)
        return sealed_write(
            out / "corrected_cstr_compliance.json",
            {
                "plan": assessed["artifact_sha256"],
                "selection": frozen["artifact_sha256"],
                "score_source": "functional rubric on corrected public training data",
                "subjects": len(rows),
                "requirements": details,
                "methods": result["methods"],
                "test_data_used": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("root", "staging", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.staging, args.output)
    print(f"Assessed {result['subjects']} corrected reactor endpoints.")
