"""Publish separate discovery inputs and analyst evidence for the nonbaseline study."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from autoformalism.benchmarks.phase_c_release import SPEC_PATHS
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.iclr_cstr_correction import CSTR, EXPECTED, NAMES
from scripts.iclr_submission_evidence import REPO, digest
from scripts.package_iclr_dynamics_evidence import pack

PUBLIC_FILES = {"specification.json", "train.json", "val.json", "proposer_prompt.txt"}
DOCS = (
    "CONTROLLED_SEARCH",
    "REFERENCE_AND_INTERVENTIONS",
    "STRESS_STUDIES",
    "IDENTIFIABILITY",
    "NONBASELINE_RUNBOOK",
    "REVIEWER_RESPONSES",
)


def public_files(stress: Path) -> tuple[dict, dict]:
    """Release only authenticated observed development inputs, plus a routing roster."""
    m = sealed_read(stress / "manifest.json")
    if (
        m["test_data_generated"] is not False
        or m["provider_calls"]
        or len(m["cells"]) != 21
    ):
        raise ValueError("expected the full public development stress design")
    archive = {}
    selectors = {}
    for cell in m["cells"]:
        key = cell["cell"]
        fields = cell["public_files"]
        if not key.isidentifier() or key in selectors or set(fields) != PUBLIC_FILES:
            raise ValueError("invalid public-only payload")
        selectors[key] = fields
        for file, sha in fields.items():
            path = stress / "public" / key / file
            if digest(path) != sha:
                raise ValueError(f"changed public data: {key}/{file}")
            archive[f"public/{key}/{file}"] = path
    anchors = ["acquisition_00", "sensor_00"]
    extra = [cell for cell in selectors if cell not in anchors]
    if len(extra) != 19 or len(m["runs"]) != 126:
        raise ValueError("incomplete public roster")
    return archive, {
        "protocol": "iclr-discovery-inputs-1",
        "public_files": selectors,
        "anchors": anchors,
        "stress": extra,
        "release_sha256": m["artifact_sha256"],
    }


def results(root: Path, reference: Path, fidelity: Path) -> dict:
    """Require actual sealed outputs; copy report files, not caches or provider logs."""
    files = {}

    def add(prefix: str, path: Path, names: list[str]) -> None:
        for name in names:
            files[f"{prefix}/{name}"] = path / name

    for folder in ("submission", "search-history", "specification"):
        path = root / folder
        for f in sorted(path.iterdir()):
            if f.suffix in {".csv", ".json", ".md", ".png"}:
                files[f"{folder}/{f.name}"] = f
    fn = root / "functional"
    add("functional", fn, ["plan.json", "functional_visits.csv"])
    plan = sealed_read(fn / "plan.json")
    for job in plan["jobs"]:
        path = fn / "jobs" / job / "score.json"
        value = sealed_read(path)
        if value["plan_sha256"] != plan["artifact_sha256"] or value["job"] != job:
            raise ValueError("foreign historical functional score")
        files[f"functional/jobs/{job}/score.json"] = path
    add(
        "reference",
        reference,
        [
            "plan.json",
            "frozen_models.json",
            "reference_structure_results.csv",
            "literature_transfer_results.csv",
        ],
    )
    ref_plan = sealed_read(reference / "plan.json")
    for path in reference.glob("fits/*/[0-9]*.json"):
        val = sealed_read(path)
        if val["plan_sha256"] != ref_plan["artifact_sha256"]:
            raise ValueError("foreign reference fit")
        files["reference/" + path.relative_to(reference).as_posix()] = path
    summary = sealed_read(fidelity / "summary.json")
    fit_plan = sealed_read(fidelity / "frozen_selection_manifest.json")
    scores = sealed_read(fidelity / "scored_rows.json")
    if (
        summary["plan_sha256"] != fit_plan["artifact_sha256"]
        or summary["scored_rows_sha256"] != scores["artifact_sha256"]
        or summary["metrics_sha256"] != public.content_sha256(scores["rows"])
        or summary["external_baselines_evaluated"] is not False
        or fit_plan["mode"] != "development"
    ):
        raise ValueError("require authenticated development-only internal results")
    add(
        "fidelity",
        fidelity,
        [
            "frozen_selection_manifest.json",
            "summary.json",
            "scored_rows.json",
            "intervention_effects.csv",
            "reference_effects.csv",
            "hidden_process_metrics.csv",
            "matched_fit_pairs.csv",
            "absolute_and_difference_trajectories.pdf",
        ],
    )
    for p in (fidelity / "references").glob("*/*.json"):
        files["fidelity/" + p.relative_to(fidelity).as_posix()] = p
    for lane, count in (("anchor-search-plan", 48), ("stress-search-plan", 114)):
        path = root / lane
        p = sealed_read(path / "plan.json")
        if sum(len(b["tasks"]) for b in p["blocks"]) != count:
            raise ValueError("incorrect production search roster")
        for name in (
            "plan.json",
            "spec_variants.json",
            "ablation_switch_manifest.json",
            "paired_ablation_results.csv",
            "search_visits.csv",
            "spec_sensitivity.csv",
        ):
            files[f"plans/{lane}/{name}"] = path / name
    files["verification/synthetic_search_smoke.json"] = (
        root / "anchor-search-plan/offline-smoke.json"
    )
    stress = root / "stress"
    for name in (
        "design.json",
        "manifest.json",
        "source_files.json",
        "generator_environment.json",
        "scaling_system_manifest.json",
        "scaling_excitation.json",
        "data_robustness_manifest.json",
    ):
        files[f"stress/{name}"] = stress / name
    for p in (stress / "evaluator").glob("device_*/system.json"):
        files["stress/" + p.relative_to(stress).as_posix()] = p
    for p in (stress / "report").iterdir():
        if p.suffix in {".json", ".csv", ".pdf"}:
            files[f"stress/report/{p.name}"] = p
    conditioned = stress / "conditioned-references"
    add(
        "stress/conditioned-references",
        conditioned,
        [
            "plan.json",
            "conditions.json",
            "frozen_models.json",
            "reference_structure_results.csv",
        ],
    )
    condition_plan = sealed_read(conditioned / "plan.json")
    for p in conditioned.glob("fits/*/[0-9]*.json"):
        if sealed_read(p)["plan_sha256"] != condition_plan["artifact_sha256"]:
            raise ValueError("foreign condition-specific reference fit")
        files[
            "stress/conditioned-references/" + p.relative_to(conditioned).as_posix()
        ] = p
    return files


def package(args) -> None:
    public_inputs, roster = public_files(args.root / "stress")
    evidence = results(args.root, args.reference, args.fidelity)
    parity = sealed_read(args.cstr_parity)
    if (
        parity["test_trajectories_generated"] is not False
        or any(
            parity["checks"][split]["fitted_csv_sha256"]
            != parity["checks"][split]["regenerated_sha256"]
            for split in ("train", "validation")
        )
    ):
        raise ValueError("corrected CSTR development parity check failed")
    evidence["verification/corrected_cstr_development.json"] = args.cstr_parity
    # The CSTR correction overlay and release for oracle fitting contain public data.
    # Only the 21 curated cell folders above are staged for proposal generation.
    corrected = (
        REPO
        / "artifacts/reference-integrity-audit-2026-09-26-v3/public-development"
        / CSTR
    )
    for name in NAMES:
        if name in EXPECTED and digest(corrected / name) != EXPECTED[name]:
            raise ValueError("CSTR historical correction identity differs")
        evidence[f"corrected-easy/{name}"] = corrected / name
    plan = sealed_read(args.reference / "plan.json")
    for cell, hashes in plan["files"].items():
        folder = args.release / cell
        for name, wanted in hashes.items():
            if digest(folder / name) != wanted:
                raise ValueError("privileged fit public inputs differ")
            evidence[f"phase-c-public/{cell}/{name}"] = folder / name
        evidence[f"phase-c-public/{cell}/proposer_prompt.txt"] = (
            folder / "proposer_prompt.txt"
        )
    spec = Path(SPEC_PATHS["cstr"])
    if spec.is_absolute() or ".." in spec.parts:
        raise ValueError("unexpected reactor spec path")
    evidence["evaluator-data/" + spec.as_posix()] = args.private / spec
    evidence["README.md"] = REPO / "docs/ICLR_2027_NONBASELINE_RESULTS.md"
    for doc in DOCS:
        name = f"ICLR_2027_{doc}.md"
        evidence[f"docs/{name}"] = REPO / "docs" / name
    evidence["claim_evidence_ledger.csv"] = REPO / "docs/ICLR_2027_CLAIM_EVIDENCE.csv"
    evidence["protocol_contract.json"] = (
        REPO / "configs/iclr_rebuttal_evaluation_v1.json"
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "roster.json").write_text(
            json.dumps(roster, indent=2, sort_keys=True) + "\n"
        )
        (tmp / "source-commit.txt").write_text(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True)
        )
        public_inputs["roster.json"] = tmp / "roster.json"
        public_inputs["source-commit.txt"] = tmp / "source-commit.txt"
        evidence["source-commit.txt"] = tmp / "source-commit.txt"
        pack(public_inputs, args.destination / "iclr-nonbaseline-public.zip")
        pack(evidence, args.destination / "iclr-nonbaseline-evidence.zip")
    print(
        f"Packaged {len(public_inputs)} public files and {len(evidence)} analyst files."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO / "output/iclr-2027-nonbaseline")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--fidelity", type=Path, required=True)
    p.add_argument(
        "--cstr-parity",
        type=Path,
        default=REPO / "output/iclr-2027-new-controls/cstr-postfreeze-check/check.json",
    )
    p.add_argument(
        "--release",
        type=Path,
        default=REPO / "artifacts/phase-c-development-final-v2/public",
    )
    p.add_argument("--private", type=Path, default=REPO / "data_raw")
    p.add_argument("--destination", type=Path, default=REPO / "output")
    package(p.parse_args())
