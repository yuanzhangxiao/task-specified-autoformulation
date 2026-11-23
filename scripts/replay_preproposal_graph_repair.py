"""Replay the seed-102 graph with its archived and pilot initialization contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.research import preproposal_cases as cases
from autoformalism.research.preproposal_graph_repair import assess_and_repair
from autoformalism.research.preproposal_selection import SelectionCase
from autoformalism.research.preproposal_topology import assess_topology

LAWS = "b405d7ef5f483f470533684bd63d96d06908faa9679ab58db6216e5130f44474"


def replay(project: Path, source: Path) -> dict:
    """Use the pinned accepted topology and the first formula reply from seed 102."""
    frozen_cases = cases.load_frozen_cases(source)
    proposal = cases._pinned(project / cases.FRESH_RECEIPT, cases.RECEIPT_DIGEST)[
        "value"
    ]
    receipts = project / cases.FRESH_DIR / "steps"
    laws = cases._pinned(receipts / "fresh-102-laws/reply.json", LAWS)["value"]
    sources = {k: v["sources"] for k, v in proposal["targets"].items()}
    sources.update({v["name"]: v["sources"] for v in proposal["additional_components"]})
    initializers = [
        {
            "state": r["state"],
            "rule": {k: r["initial"][k] for k in ("mode", "scope")},
            "justification": "Archived initializer scope.",
        }
        for r in laws["latent_initials"]
    ]
    original = {
        "sources": sources,
        "hypothesis": proposal["hypothesis"],
        "latent_initializers": initializers,
    }
    # Preserve the original in the report. Compare wiring using the pilot's
    # time-zero-only contract and the same shared-constant control as the v1 audit.
    common = {
        **original,
        "latent_initializers": [
            {
                **r,
                "rule": {"mode": "value", "scope": "shared_training"},
                "justification": "Controlled shared initial value for this replay.",
            }
            for r in initializers
        ],
    }
    results = {}
    for key in (cases.T2_CASE, cases.T2_GAP_CASE):
        frozen = frozen_cases[key]
        case = SelectionCase.model_validate(cases.selection_case(frozen))
        results[key] = {
            "archived_initializer_assessment": assess_topology(
                case, None, original, frozen
            ),
            "shared_initial_control": assess_and_repair(case, common, frozen),
        }
    return {
        "original": original,
        "topology_receipt": cases.RECEIPT_DIGEST,
        "formula_receipt": LAWS,
        "cases": results,
        "optimizer_calls": 0,
        "provider_calls": 0,
        "test_data_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = replay(args.project, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, output)
    for case_id, analysis in output["cases"].items():
        r = analysis["shared_initial_control"]
        print(
            case_id, r["status"], r["checked"], "checked,", len(r["repairs"]), "passing"
        )
