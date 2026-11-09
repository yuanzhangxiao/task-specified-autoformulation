"""Join historical cohorts, fitted graphs, scores and provenance."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_active_history as active
from scripts.iclr_search_history import INVENTORY, assessed_visits
from scripts.iclr_submission_evidence import (
    ARCHIVE,
    PLAN,
    REPO,
    digest,
    load_archive,
    table,
)

EVIDENCE = REPO / "output/iclr-2027-nonbaseline"
EARLIER = REPO / "output/iclr-2027-rebuttal-evidence"
FIDELITY = REPO / "output/iclr-2027-new-controls/fidelity-release"
REFERENCE = REPO / "output/iclr-2027-new-controls/reference-final"
OUT = REPO / "output/iclr-local-completion/historical"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def export_csv(path, rows):
    """Serialize nested identifiers and lists as JSON inside CSV cells."""
    table(
        path,
        [
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in r.items()
            }
            for r in rows
        ],
    )


def registry(archive_path, evidence, earlier, fidelity, reference):
    """Register the release, receipt and selection for each cohort."""
    records = []

    def row(id, release, source, receipt, schedule, planned, unit, selection, status):
        records.append(
            {
                "cohort_id": id,
                "release": release,
                "source_receipt": source,
                "source_sha256": receipt,
                "stage_and_split": schedule,
                "planned_or_eligible": planned,
                "denominator_unit": unit,
                "selection": selection,
                "status": status,
            }
        )

    row(
        "historical-six",
        "Phase B 9 submitted cases",
        "outputs/all-benchmarks-mad-20260925/all-benchmarks-validation-data.zip",
        digest(archive_path),
        "six visits, pre-pruning, validation",
        160,
        "lineages (144 main and 16 secondary); 2 seeds per prompt",
        "training fit, validation selection",
        "reconstructed",
    )
    manifest = sealed_read(earlier / "comparisons/manifest.json")
    row(
        "historical-t1-pairs",
        "Phase B T1 named and obfuscated",
        "comparisons/manifest.json",
        manifest["artifact_sha256"],
        "round-12 frozen retrospective paired schedules",
        manifest["planned_endpoints"],
        "planned model endpoints across methods/presentations",
        "archived endpoints; assay evaluated after selection",
        "measured",
    )
    m = sealed_read(earlier / "meal-appearance/plan.json")
    row(
        "historical-t1-clamp",
        "Phase B T1",
        "meal-appearance/plan.json",
        m["artifact_sha256"],
        "retrospective meal-flux clamp",
        len(m["models"]),
        "eligible frozen endpoints in clamp plan",
        "archived endpoints; schedule-specific conditional diagnostic",
        "measured",
    )
    m = sealed_read(earlier / "t2-action/summary.json")
    row(
        "historical-t2-action",
        "Phase B T2 hard",
        "t2-action/summary.json",
        m["artifact_sha256"],
        "retrospective fasting pulses",
        len(m["models"]),
        "archived endpoint summaries",
        "frozen historical models",
        "measured",
    )
    m = sealed_read(reference / "plan.json")
    row(
        "privileged-clean-reference",
        "Phase C 4 canonical-structure problems",
        "reference/plan.json",
        m["artifact_sha256"],
        "development training and validation",
        m["starts"] * len(m["files"]),
        "optimizer starts; four validation-selected endpoints",
        "train fit; validation selection",
        "measured",
    )
    m = sealed_read(evidence / "stress/conditioned-references/plan.json")
    row(
        "privileged-conditioned-reference",
        "Phase C altered acquisition",
        "stress/conditioned-references/plan.json",
        m["artifact_sha256"],
        "development training and validation",
        m["starts"] * len(m["files"]),
        "optimizer starts; masked reactor and noise T2",
        "train fit; validation selection",
        "measured",
    )
    m = sealed_read(fidelity / "frozen_selection_manifest.json")
    for id, stage in (
        ("probe-calibration", "training schedules and simulator labels in evaluator"),
        ("fidelity-development", "novel development intervention schedules"),
    ):
        row(
            id,
            "historical Phase B and privileged Phase C rows with separate case IDs",
            "fidelity/frozen_selection_manifest.json",
            m["artifact_sha256"],
            stage,
            len(m["roster"]),
            "frozen subjects, 16 historical T2 and 4 privileged",
            "no fitting of candidate dynamics by assay",
            "measured",
        )
    for id, p in (
        ("phase-c-anchors", "anchor-search-plan"),
        ("phase-c-extra-stress", "stress-search-plan"),
    ):
        m = sealed_read(evidence / p / "plan.json")
        n = sum(len(b["tasks"]) for b in m["blocks"])
        row(
            id,
            "Phase C acquisition, sensing and procedural controls",
            "plans/" + p + "/plan.json",
            m["artifact_sha256"],
            "search and post-freeze confirmation",
            n,
            "new algorithm endpoints; seeds 0,1,2",
            "training fit; validation selection; independent freeze",
            "pending",
        )
        if id == "phase-c-anchors":
            row(
                "fidelity-confirmation",
                "Phase C hard anchors",
                "plans/" + p + "/plan.json",
                m["artifact_sha256"],
                "independent prespecified confirmation",
                n,
                "planned new anchor endpoints plus separately labeled controls",
                "require same-release sealed endpoint roster",
                "pending",
            )
    m = sealed_read(evidence / "stress/manifest.json")
    row(
        "phase-c-stress-report",
        "Phase C stress release",
        "stress/manifest.json",
        m["artifact_sha256"],
        "independent post-freeze observations",
        len(m["runs"]),
        "endpoints including 12 shared anchor runs",
        "pair full and fixed arms within each cell and seed",
        "pending",
    )
    row(
        "corrected-cstr-historical",
        "Phase B corrected named CSTR-easy",
        "original component plan (parent only)",
        PLAN,
        "new corrected search and corrected held-out test after freeze",
        16,
        "four submitted arms x two prompts x two original seeds",
        "train and validate on corrected schedules",
        "only corrected development parity measured; final rerun pending",
    )
    return records


def corrections():
    return [
        {
            "location": a,
            "printed_or_inferred_claim": b,
            "actual_provenance": c,
            "supported_replacement": d,
            "affected_conclusion": e,
            "remaining": f,
        }
        for a, b, c, d, e, f in (
            (
                "Tables 2-3",
                "internal method test NMSE",
                "six visits, before pruning, validation",
                (
                    "label .397/.538/.551/.491 as validation; "
                    "available 35/35/36/36 out of 36 per arm"
                ),
                "common test ranking unavailable",
                "frozen same-release evaluation",
            ),
            (
                "Experiment protocol",
                "three seeds per case and fourteen revisions for printed NMSE",
                (
                    "two algorithm seeds x two prompt variants "
                    "and six visits at printed cutoff"
                ),
                (
                    "distinguish planned 15 visits from six-visit "
                    "snapshot; give prompt-stratified scores"
                ),
                "experimental unit and sample size change",
                "none for reconstruction",
            ),
            (
                "Tables 2-3 compliance",
                "88.9% and other arms form one functional comparison",
                "Brief-only graph check and mixed functional endpoints/prompt policies",
                (
                    "Full method functional 58.3% with Full "
                    "prompt or 72.2% with Brief-only, pre-pruning"
                ),
                "no common compliance ranking from the mixed table",
                "recover raw scorer receipts or rescore matched cohort",
            ),
            (
                "Results text",
                "85.0% compliance",
                "untraced",
                "remove percentage unless traced",
                "numerator and endpoint unknown",
                "original supporting source",
            ),
            (
                "CSTR benchmark",
                (
                    "reported reactor label and historical "
                    "results describe corrected hard task"
                ),
                (
                    "named easy task used affected numerical "
                    "export; 11 train, 3 validation, 3 test schedules differ"
                ),
                "identify original release; report corrected CSTR as separate cohort",
                "changing reference values affects training and evaluation",
                "16-lineage corrected search and post-freeze test",
            ),
            (
                "Latent recovery",
                "compliant pathway identifies physiological mechanism",
                (
                    "historical learned T2 direct evaluator "
                    "correspondence absent in published panel"
                ),
                (
                    "give paired target-response and calibrated "
                    "readout metrics by intervention"
                ),
                "structural compliance alone cannot establish mechanism recovery",
                "independent confirmation",
            ),
            (
                "Worked search lineage",
                "a hidden-memory pathway appeared only at the second visit",
                (
                    "visit 1 X already participates in disposal; "
                    "visit 2 introduces a replacement driving state"
                ),
                (
                    "show dynamic paths, orphaned X, operative "
                    "sign and trial versus retained lineage"
                ),
                "additional declared state need not be active",
                "verbatim historical critic messages unavailable",
            ),
            (
                "Canonical T1 transfer",
                "perturbed validation error isolates a wrong equation family",
                (
                    "seven of nine coordinates saturated original "
                    "fitted bounds in every start"
                ),
                (
                    "condition inference on the unchanged model, "
                    "bounds and optimizer starts"
                ),
                "limits confound structural attribution",
                "quantify broader-bounds sensitivity",
            ),
        )
    ]


def functional_roles(folder, seeds, pairs):
    """Compare confirmed public roles on the same fitted models at visit six."""
    plan = sealed_read(folder / "plan.json")
    scores = {}
    for r in plan["rows"]:
        if r["round"] != 5 or r["job"] is None:
            continue
        path = folder / "jobs" / r["job"] / "score.json"
        if not path.exists():
            continue
        record = sealed_read(path)
        if (
            record["plan_sha256"] != plan["artifact_sha256"]
            or record["job"] != r["job"]
        ):
            raise ValueError("wrong operational score receipt")
        scores[r["task"]] = (
            r["result_sha256"],
            {f["id"]: f["status"] for f in record["findings"]},
        )
    for r in seeds:
        if r["task"] in scores:
            receipt, roles = scores[r["task"]]
            if receipt != r["source_result_sha256"]:
                raise ValueError("operational score attached to a different visit")
            r["role_status"] = roles
    for pair in pairs:
        roles = [scores.get(pair[k]) for k in ("seed0_task", "seed1_task")]
        if all(roles):
            a, b = (r[1] for r in roles)
            if set(a) != set(b):
                raise ValueError("public requirements differ across paired seeds")
            pair["operational_status_agreement"] = sum(a[k] == b[k] for k in a) / len(a)
            pair["confirmed_roles_jaccard"] = active.overlap(
                [k for k, v in a.items() if v == "pass"],
                [k for k, v in b.items() if v == "pass"],
            )


def fixed_plot(rows, out):
    """Plot fixed-run cohorts, displaying both horizon and denominator."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5), layout="constrained")
    axes = fig.subplots(2, 2, sharex="col")
    for col, h in enumerate((6, 15)):
        panel = axes[:, col]
        for prompt in ("full", "brief_only"):
            data = [
                r
                for r in rows
                if r["arm"] == "c1v1s1" and r["prompt"] == prompt and r["horizon"] == h
            ]
            if not data:
                continue
            n = data[0]["fixed_runs"]
            for ax, key in zip(
                panel, ("median_validation_nmse", "mean_confirmed"), strict=True
            ):
                ax.plot(
                    [r["visit"] for r in data],
                    [np.nan if r[key] is None else r[key] for r in data],
                    marker=".",
                    label=f"{prompt} ({n}/{data[0]['planned_runs']})",
                )
        panel[0].set(
            title=f"Joint measurements at all {h} visits",
            ylabel="Median validation NMSE",
        )
        panel[0].legend(fontsize=8)
        panel[1].set(
            xlabel="Visit", ylabel="Mean functional pass fraction", ylim=(-0.05, 1.05)
        )
    fig.savefig(out / "fixed_cohort_learning.png", dpi=150)


def run(
    out: Path,
    evidence=EVIDENCE,
    earlier=EARLIER,
    fidelity=FIDELITY,
    reference=REFERENCE,
    archive=ARCHIVE,
    inventory=INVENTORY,
):
    search, _ = load_archive(archive)
    inv = sealed_read(inventory)
    snapshot = sealed_read(evidence / "search-history/snapshot.json")
    if (
        snapshot["inventory"] != inv["artifact_sha256"]
        or inv["plan_sha256"] != PLAN
        or snapshot["archive"] != digest(archive)
        or inv["test_data_opened"] is not False
    ):
        raise ValueError("foreign historical source")
    scores = assessed_visits(evidence / "functional", search["rows"])
    inventory_rows = {
        r["task"]: r for r in read_csv(evidence / "submission/run_inventory.csv")
    }
    rows, pairs, traces = active.histories(inv, search["rows"], inventory_rows, scores)
    functional_roles(evidence / "functional", rows, pairs)
    curves = [
        r for h in (6, 15) for r in active.fixed_curves(search["rows"], scores, h)
    ]
    behavior = active.behavioral_pairs(fidelity)
    cohorts = registry(archive, evidence, earlier, fidelity, reference)
    corrected = corrections()
    claims = read_csv(REPO / "docs/ICLR_2027_CLAIM_EVIDENCE.csv")
    for c in claims:
        c["scope"] = c["limit"]
    claims.extend(
        [
            {
                "reviewer": "7pHK",
                "question": "Are learned coordinates physical readouts?",
                "measurement": (
                    "384 learned direct-readout rows marked "
                    "physical_correspondence_unavailable"
                ),
                "evidence_path": "fidelity/hidden_process_metrics.csv",
                "status": "measured",
                "scope": "historical development assay, 16 T2-hard models",
                "limit": "scalar probes measure decodability",
            },
            {
                "reviewer": "Ewor",
                "question": "How do the two seeds compare behaviorally?",
                "measurement": "paired response differences by schedule and target",
                "evidence_path": "seed_behavior.csv",
                "status": "measured",
                "scope": "8 archived T2-hard arms, both seeds",
                "limit": (
                    "descriptive pairwise curves, no replicate confidence interval"
                ),
            },
        ]
    )
    with public._lock(out):
        transfer = sealed_read(out / "transfer_summary.json")
        cohorts.append(
            {
                "cohort_id": "bounded-transfer-sensitivity",
                "release": "Phase C perturbed T1-easy",
                "source_receipt": "transfer_plan.json",
                "source_sha256": transfer["plan"],
                "stage_and_split": "development training and validation",
                "planned_or_eligible": 6,
                "denominator_unit": (
                    "three start indices under two initialization policies"
                ),
                "selection": "minimum validation NMSE per profile",
                "status": "measured",
            }
        )
        corrected[-1]["remaining"] = (
            "six expanded-bound development fits "
            "completed; no structural impossibility inference"
        )
        tables = {
            "cohort_registry": cohorts,
            "manuscript_corrections": corrected,
            "claims_status": claims,
            "historical_per_seed": rows,
            "paired_active_stability": pairs,
            "visit_structure_edits": traces,
            "fixed_cohort_learning": curves,
            "seed_behavior": behavior,
            "worked_lineage": active.worked_lineage(inv, traces),
        }
        for name, data in tables.items():
            export_csv(out / f"{name}.csv", data)
        fixed_plot(curves, out)
        graph = [p for p in pairs if p["comparable"]]
        return sealed_write(
            out / "historical_summary.json",
            {
                "protocol": "iclr-local-history-1",
                "source_inventory": inv["artifact_sha256"],
                "archive": digest(archive),
                "functional_source": sealed_read(evidence / "functional/plan.json")[
                    "artifact_sha256"
                ],
                "transfer_summary": transfer["artifact_sha256"],
                "source_scripts": [
                    digest(Path(__file__)),
                    digest(Path(active.__file__)),
                ],
                "rows": {name: len(data) for name, data in tables.items()},
                "table_digests": {name: digest(out / f"{name}.csv") for name in tables},
                "comparable_pairs": len(graph),
                "equal_active_dimension": sum(
                    p["same_active_dimension"] for p in graph
                ),
                "structure_coverage": dict(
                    Counter(r["structure_status"] for r in rows)
                ),
                "operational_per_seed_covered": sum(
                    r["functional_status"] == "measured" for r in rows
                ),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
