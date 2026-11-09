"""Verify and bundle the frozen latent-dynamics rebuttal evidence for handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from autoformalism.rebuttal.prefit_replay import sealed_read

PROJECT = Path(__file__).resolve().parents[1]


def check_csv_fields(saved: dict, row: dict) -> None:
    """Compare CSV serialization with the corresponding sealed numerical record."""
    for field, value in row.items():
        if field == "delta_flux":
            continue
        if saved.get(field) != ("" if value is None else str(value)):
            raise ValueError(f"CSV and sealed record disagree on {field}")


def verify(root: Path) -> None:
    """Require the frozen rosters, compatible plans and every planned result row."""
    subjects = sealed_read(root / "subjects/models.json")
    pairs = sealed_read(root / "subjects/matched-pairs.json")
    t2 = sealed_read(root / "t2-action/summary.json")
    t2plan = sealed_read(root / "t2-action/plan.json")
    if not (
        pairs["subject_bundle_sha256"]
        == subjects["artifact_sha256"]
        == t2plan["source_bundle_sha256"]
        and t2["plan_sha256"] == t2plan["artifact_sha256"]
    ):
        raise ValueError("T2 roster, matching or evaluation identity differs")
    planned = {(r["method"], r["repetition"]) for r in subjects["rows"]}
    scored = {(r["method"], r["repetition"]) for r in t2["models"]}
    if len(planned) != 28 or planned != scored or len(t2["models"]) != 28:
        raise ValueError("T2 cohort incomplete")
    with (root / "t2-action/metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    keys = {
        (r["method"], int(r["repetition"]), r["case"], r["endpoint"]) for r in metrics
    }
    expected = {
        (m, n, c, e)
        for m, n in planned
        for c in ("early_short", "late_long")
        for e in ("Gp", "I", "X_from_latent", "X_from_generated_I")
    }
    if keys != expected or len(metrics) != len(expected):
        raise ValueError("T2 pulse/endpoint grid is incomplete")
    t2_scores = {
        (s["method"], s["repetition"], s["case"], s["endpoint"]): s
        for m in t2["models"]
        for s in m["scores"]
    }
    for r in metrics:
        check_csv_fields(
            r, t2_scores[r["method"], int(r["repetition"]), r["case"], r["endpoint"]]
        )
    t1 = sealed_read(root / "meal-appearance/plan.json")
    with (root / "meal-appearance/scores.csv").open() as f:
        flux_rows = list(csv.DictReader(f))
    expected = {
        (m, c) for m in t1["models"] for c in t1["meal_schedules"] if c != "fasting"
    }
    if {(r["model_id"], r["case"]) for r in flux_rows} != expected or len(
        flux_rows
    ) != len(expected):
        raise ValueError("T1 flux grid is incomplete")
    for model_id in t1["models"]:
        record = sealed_read(root / "meal-appearance/models" / (model_id + ".json"))
        if (
            record["model_id"] != model_id
            or record["plan_sha256"] != t1["artifact_sha256"]
        ):
            raise ValueError("T1 flux model belongs to another plan")
        for r in flux_rows:
            if r["model_id"] == model_id:
                check_csv_fields(r, record["cases"][r["case"]])
    comparison = sealed_read(root / "comparisons/manifest.json")
    with (root / "comparisons/paired_responses.csv").open() as f:
        if len(list(csv.DictReader(f))) != comparison["paired_rows"]:
            raise ValueError("T1 response grid is incomplete")


def pack(files: dict[str, Path], output: Path) -> None:
    """Publish the ZIP atomically with deterministic members and SHA-256 checksums."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        target = Path(tmp) / "bundle.zip"
        sums = {}
        with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
            for name, path in sorted(files.items()):
                data = path.read_bytes()
                sums[name] = hashlib.sha256(data).hexdigest()
                archive.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)
            contents = "".join(f"{digest}  {name}\n" for name, digest in sums.items())
            archive.writestr(
                ZipInfo("SHA256SUMS"), contents, compress_type=ZIP_DEFLATED
            )
        with ZipFile(target) as archive:
            for name, digest in sums.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f"archived data mismatch: {name}")
        target.replace(output)


def package(root: Path, output: Path) -> None:
    """Bundle the scientific results, analysis sources and frozen inputs."""
    verify(root)
    files = {
        "README.md": PROJECT / "docs/ICLR_2027_DYNAMICS_EVIDENCE.md",
        "inputs/t1-models.json": PROJECT
        / "artifacts/t1-curves-round12-2026-09-22/models.json",
        "provenance/validation_ablation.tex": PROJECT
        / "outputs/all-benchmarks-mad-20260925/ablation_table.tex",
        "provenance/external_test_nmse.tex": PROJECT
        / "output/results-package-v1-failure-aware/TABLE_VALUES.tex",
    }
    report_files = {
        "comparisons": (
            "README.md",
            "endpoints.csv",
            "manifest.json",
            "paired_meal_responses.png",
            "paired_responses.csv",
            "perturbed_meal_solver_verified.csv",
        ),
        "meal-appearance": (
            "endpoints.csv",
            "meal-appearance.pdf",
            "meal-appearance.png",
            "plan.json",
            "reference.json",
            "scores.csv",
        ),
        "subjects": ("README.md", "lineage.json", "matched-pairs.json", "models.json"),
        "t2-action": (
            "action-response.pdf",
            "action-response.png",
            "metrics.csv",
            "plan.json",
            "summary.json",
        ),
    }
    for folder, names in report_files.items():
        for name in names:
            files[f"{folder}/{name}"] = root / folder / name
    t1plan = sealed_read(root / "meal-appearance/plan.json")
    for model in t1plan["models"]:
        name = f"meal-appearance/models/{model}.json"
        files[name] = root / name
    t2plan = sealed_read(root / "t2-action/plan.json")
    for case in t2plan["cases"]:
        name = f"t2-action/reference/{case['id']}.json"
        files[name] = root / name
        for i, row in enumerate(t2plan["rows"]):
            if row["status"] == "ready":
                name = f"t2-action/subjects/{i:03d}/{case['id']}.json"
                files[name] = root / name
    for name in (
        "analyze_t2_hard_selection",
        "compare_t1_rebuttal_effects",
        "evaluate_t1_meal_appearance",
        "evaluate_t2_latent_dynamics",
        "package_iclr_dynamics_evidence",
        "plot_t1_rebuttal_effects",
        "plot_t2_latent_dynamics",
        "prepare_t2_hard_subjects",
        "report_t1_meal_appearance",
    ):
        files[f"scripts/{name}.py"] = PROJECT / f"scripts/{name}.py"
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    )
    with tempfile.TemporaryDirectory() as staging:
        revision_file = Path(staging) / "source-commit.txt"
        revision_file.write_text(revision)
        files["source-commit.txt"] = revision_file
        pack(files, output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    package(args.root, args.output)
