#!/usr/bin/env python3
"""Replay saved declarations and export unresolved requirements without judging."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autoformalism.expressions import ModelValidationError
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.revision_decision import parameter_aliases
from autoformalism.schemas import CandidateModel
from autoformalism.search import review_revision_v4 as edits


def declarations(bundle: dict, raw: dict) -> list[dict]:
    """Classify saved declarations from actual parent identity, not name spelling."""
    model = CandidateModel.model_validate(bundle["initialization"]["base_candidate"])
    existing = {p.name for p in model.parameters}
    inverse = {v: k for k, v in parameter_aliases(model).items()}
    items = [
        (e["component"], p)
        for e in raw.get("equations", [])
        for p in e.get("parameters", [])
    ]
    items += [("whole_revision", p) for p in raw.get("new_parameters", [])]
    return [
        {
            "component": component,
            "name": p["name"],
            "role": p.get("role"),
            "inherited": inverse.get(p["name"], p["name"]) in existing,
        }
        for component, p in items
    ]


def audit(root: Path) -> dict:
    plan = io.verify(root)
    if plan["protocol"] != io.PARAMETER_PROTOCOL:
        raise ValueError("requires parameter-declaration continuation")
    ledger = plan["continuation"]
    source = Path(ledger["source_root"])
    original = io.verify(source, execution=False)
    if original["artifact_sha256"] != ledger["source_plan_sha256"]:
        raise ValueError("source plan changed")
    records, requirements, terminal = [], [], Counter()
    for task in plan["tasks"]:
        for index in range(1, ledger["source_phase_round"] + 1):
            path = io.round_path(source, task, index) / "proposal.json"
            if not path.exists():
                continue
            proposal = sealed_read(path)
            parent = io.read_round(source, task, index - 1)
            if parent is None or proposal["parent_sha256"] != parent["artifact_sha256"]:
                raise ValueError("saved proposal parent differs")
            attempts = proposal.get("attempts", [])
            if proposal["status"] == "revision_failed":
                terminal[
                    (attempts[-1].get("feedback") or {}).get(
                        "code", "delivery_or_budget"
                    )
                    if attempts
                    else "delivery_or_budget"
                ] += 1
            for i, attempt in enumerate(attempts):
                record = {
                    "task": task["task_id"],
                    "source_phase_round": index,
                    "round": index
                    + original.get("continuation", {}).get("source_round", 0),
                    "attempt": i,
                    "source_accepted": attempt["accepted"],
                    "proposal_sha256": proposal["artifact_sha256"],
                    "previous_feedback": attempt.get("feedback"),
                }
                try:
                    selected = parent["selected"]
                    bundle, packet = selected["bundle"], selected["packet"]
                    raw = attempt.get("raw")
                    if raw is None:
                        raise ValueError(
                            "no parsed content; delivery/schema failure retained"
                        )
                    record["declarations"] = declarations(bundle, raw)
                    if original["protocol"] == io.CONTENT_PROTOCOL:
                        from audit_review_continuation import convert

                        raw = convert(raw, bundle["context"]["targets"][0])
                    if original["protocol"] != io.PARAMETER_PROTOCOL:
                        raw = edits.migrate_saved(raw)
                    result = edits.apply_edits(bundle, packet, raw)
                    if result["bundle"] is not None:
                        certificate = pipeline.certificates(
                            result["bundle"], original["cells"][task["cell"]], task
                        )
                        record["requirement_findings"] = pipeline._certificate_feedback(
                            certificate, task
                        )
                        if not certificate["eligible_for_development_selection"]:
                            raise ValueError(
                                "public requirement or ablation check still fails"
                            )
                    record.update(
                        status="content_valid",
                        outcome=result["outcome"],
                        parameter_audit=result["provenance"][
                            "parameter_declaration_audit"
                        ],
                    )
                except (ValueError, KeyError, TypeError, ModelValidationError) as error:
                    record.update(
                        status="still_blocked",
                        error=str(error)[:2000],
                        details=getattr(error, "details", {}),
                    )
                records.append(record)
        selected = io.read_round(root, task, 0)["selected"]
        if selected is None:
            continue
        certificate = selected["certificate"]
        unresolved = [
            p
            for p in certificate["mechanisms"]["predicates"]
            if p["status"] != "satisfied"
        ]
        if unresolved or not certificate["eligible_for_development_selection"]:
            bundle = selected["bundle"]
            requirements.append(
                {
                    "task": task,
                    "predicates": unresolved,
                    "target_predicates": [
                        p
                        for p in certificate["targets"]["predicates"]
                        if p["status"] != "satisfied"
                    ],
                    "public_requirements": plan["cells"][task["cell"]][
                        "mechanism_spec"
                    ]["required_mechanisms"],
                    "candidate": bundle["candidate"],
                    "context": bundle["context"],
                    "initialization_plan": bundle["initialization"]["plan"],
                    "fitted_parameters": selected["fit"]["parameters"],
                    "verdict": "manual_review_required_no_automatic_certification",
                }
            )
    result = sealed_write(
        root / "parameter_response_audit.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "records": records,
            "status_counts": dict(Counter(r["status"] for r in records)),
            "source_terminal_errors": dict(terminal),
            "rejected_then_valid": sum(
                not r["source_accepted"] and r["status"] == "content_valid"
                for r in records
            ),
            "llm_calls": 0,
            "fitting_calls": 0,
            "test_data_opened": False,
        },
    )
    sealed_write(
        root / "requirement_review.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "cases": requirements,
            "test_data_opened": False,
            "automatic_certificate_changes": False,
        },
    )
    lines = [
        "# Unresolved requirement review",
        "",
        "Diagnostic export only. No new scientific verdict, benchmark edit, "
        "or fitter change.",
        "The prediction-only control (internal ID no_spec) omits scientific "
        "requirements; compare task satisfaction alongside NMSE.",
        "",
    ]
    for case in requirements:
        lines += [
            "## " + case["task"]["task_id"],
            "",
            json.dumps(case["predicates"]),
            "",
        ]
        for requirement in case["public_requirements"]:
            lines += [
                "Public requirement: "
                + str(requirement.get("public_requirement") or requirement)
            ]
        candidate = case["candidate"]
        lines += ["", "```text"]
        lines += [e["state"] + "' = " + e["rhs"] for e in candidate["state_equations"]]
        lines += [p["name"] + " = " + p["expression"] for p in candidate["processes"]]
        lines += [
            m["channel"] + " observed = " + m["expression"]
            for m in candidate["observation_mappings"]
        ]
        lines += ["```", ""]
        if any(
            p.get("mechanism_id") == "controlled_balance" for p in case["predicates"]
        ):
            lines += [
                "Review the distinct feed transport, reaction heat generation, "
                "and jacket exchange terms for controlled_balance. "
                "Do not invent a driver to satisfy a generic graph check.",
                "",
            ]
    (root / "REQUIREMENT_REVIEW.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    value = audit(parser.parse_args().root)
    print(json.dumps({k: v for k, v in value.items() if k != "records"}, indent=2))
