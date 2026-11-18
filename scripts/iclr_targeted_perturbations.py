"""Stage, prepare and export the two approved 20B perturbation searches."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.phase_c_release import load_public_cell
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import iclr_controlled_search as search
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts.iclr_anchor_fidelity import verify_controlled_bundle
from scripts.iclr_controlled_receipts import export as export_receipts
from scripts.iclr_t1_perturbed_full import ANCHOR_PLAN, CONFIG, FILES, SEEDS
from scripts.iclr_t1_perturbed_full import CELL as T1

PROTOCOL = "iclr-targeted-perturbations-1"
CSTR = "phase_c_cstr_controlled_reactor_mechanism_perturbed_named_hard_reset_v1"
CELLS = [CSTR, T1]
CONFIG_PATH = search.io.REPO / "configs/iclr_targeted_perturbations_v1.json"
ARCHIVE_PREFIX = "targeted-perturbation-handoff-20261117/"


def config() -> dict:
    """Read the fixed input identities, without the private perturbation formula."""
    return json.loads(CONFIG_PATH.read_text())


def public_files(directory: Path) -> dict:
    """Hash precisely the four discovery-visible files, never traverse test data."""
    return {name: search.file_digest(directory / name) for name in FILES}


def verify_public(directory: Path) -> dict:
    """Reject changed data, prompts, channel roles or seals before launching."""
    expected = config()["public_files"]
    for cell in CELLS:
        if public_files(directory / cell) != expected[cell]:
            raise ValueError(f"targeted public files differ: {cell}")
        spec, _, _ = load_public_cell(directory / cell)
        prompt = (directory / cell / "proposer_prompt.txt").read_text()
        if spec["public_prompt"].rstrip("\n") != prompt.rstrip("\n"):
            raise ValueError("specification and proposer prompt disagree")
    return expected


def stage(archive: Path, original_t1: Path, destination: Path) -> dict:
    """Copy the approved inputs; change only T1's explicitly authorized prompt."""
    cfg = config()
    if search.file_digest(archive) != cfg["handoff_sha256"]:
        raise ValueError("wrong targeted handoff archive")
    if public_files(original_t1) != FILES:
        raise ValueError("original T1 public files differ from run 2353678")
    original = read_seal(original_t1 / "specification.json")
    with zipfile.ZipFile(archive) as z:
        # Fixed allowlist: no extraction of README, private-law audit or test files.
        payloads = {
            name: z.read(ARCHIVE_PREFIX + f"public/{CSTR}/{name}") for name in FILES
        }
        prompt = z.read(ARCHIVE_PREFIX + "t1_exchange_required/proposer_prompt.txt")
    old_prompt = (original_t1 / "proposer_prompt.txt").read_text()
    expected_prompt = old_prompt.replace(
        "following task-required mechanism:", "following task-required mechanisms:", 1
    ).replace(
        "\n\nThe dimensionality and internal representation of this mechanism is",
        "\n- "
        + search.controls.T1_EXCHANGE_CLAUSE
        + "\n\nThe dimensionality and internal representation of these mechanisms are",
        1,
    )
    if prompt.decode() != expected_prompt:
        raise ValueError("T1 prompt changes more than the approved section-A edit")
    with public._lock(destination):
        for cell, data in (
            (CSTR, payloads),
            (
                T1,
                {
                    "train.json": (original_t1 / "train.json").read_bytes(),
                    "val.json": (original_t1 / "val.json").read_bytes(),
                    "proposer_prompt.txt": prompt,
                },
            ),
        ):
            folder = destination / cell
            folder.mkdir(exist_ok=True)
            for name, value in data.items():
                path = folder / name
                if path.exists() and path.read_bytes() != value:
                    raise ValueError(f"existing staged input differs: {path}")
                if not path.exists():
                    temp = path.with_suffix(path.suffix + ".tmp")
                    temp.write_bytes(value)
                    temp.replace(path)
        seal(
            destination / T1 / "specification.json",
            {**original, "public_prompt": prompt.decode()},
        )
        files = verify_public(destination)
    return {"public_files": files, "test_data_opened": False}


def prepare(root: Path, inputs: Path, anchor: Path, gate: Path) -> dict:
    """Reuse the anchor method and calibration for six independently fresh starts."""
    original = sealed_read(anchor / "plan.json")
    if original["artifact_sha256"] != ANCHOR_PLAN:
        raise ValueError("expected the original anchor plan")
    if original["runtime"] != public._runtime():
        raise ValueError(
            f"Anchor fitting runtime required: {original['runtime']}; "
            f"found {public._runtime()}. Use the anchor activation script."
        )
    expected = verify_public(inputs)
    auth = critic.checked_authorization(anchor, original)
    if (
        sealed_read(gate / "plan.json")["artifact_sha256"]
        != auth["calibration_plan_sha256"]
        or content_hash(json.loads((gate / "summary.json").read_text()))
        != auth["summary_sha256"]
    ):
        raise ValueError("critic calibration differs from the anchor")
    for protected in (inputs, anchor, gate):
        if root.resolve().is_relative_to(protected.resolve()) or (
            protected.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("use disjoint campaign and input directories")
    with public._lock(root), TemporaryDirectory() as tmp:
        staged = Path(tmp) / "search"
        base = search.prepare(staged, inputs, CELLS, ["full"], SEEDS, 15, True, True)
        if any(base[k] != original[k] for k in CONFIG):
            raise ValueError("search or fitting settings differ from the anchor")
        roles = (
            "arm",
            "treatment",
            "scientific_verifier",
            "critic",
            "shared_processes",
            "secondary",
        )
        full_tasks = [
            t
            for b in original["blocks"]
            for t in b["tasks"]
            if t["treatment"] == "full"
        ]
        if not full_tasks or any(
            task[k] != full_tasks[0][k]
            for b in base["blocks"]
            for task in b["tasks"]
            for k in roles
        ):
            raise ValueError("Full treatment switches differ from the anchor")
        for cell in CELLS:
            saved = sealed_read(staged / "cells" / f"{cell}.json")
            if saved["files"] != expected[cell]:
                raise ValueError("staged cell hashes differ")
            sealed_write(
                root / "cells" / f"{cell}.json",
                {k: v for k, v in saved.items() if k != "artifact_sha256"},
            )
        for filename in ("spec_variants.json", "ablation_switch_manifest.json"):
            saved = sealed_read(staged / filename)
            saved.pop("artifact_sha256")
            if filename == "ablation_switch_manifest.json":
                saved["initial_construction"] = (
                    "Full builds a fitted start at visit zero."
                )
            sealed_write(root / filename, saved)
        base.pop("artifact_sha256")
        plan = sealed_write(
            root / "plan.json",
            {
                **base,
                "switch_manifest_sha256": sealed_read(
                    root / "ablation_switch_manifest.json"
                )["artifact_sha256"],
                "require_fitted_full_start": True,
                "reference_plan_sha256": ANCHOR_PLAN,
                "targeted_protocol": PROTOCOL,
                "targeted_config_sha256": search.file_digest(CONFIG_PATH),
                "study": "Full 20B: perturbed CSTR and exchange-required perturbed T1.",
                "maximum_parallel_blocks": 6,
            },
        )
        critic.bind_calibration(root, plan, gate / "plan.json", gate / "summary.json")
    return plan


def verify(root: Path) -> dict:
    """Check execution identity and the complete frozen scientific roster."""
    plan = sealed_read(root / "plan.json")
    if (
        plan.get("targeted_protocol") != PROTOCOL
        or plan.get("targeted_config_sha256") != search.file_digest(CONFIG_PATH)
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["arms"] != ["full"]
        or plan["visits"] != 15
        or plan["seeds"] != SEEDS
        or not plan.get("require_fitted_full_start")
        or [(b["cell"], b["seed"]) for b in plan["blocks"]]
        != [(c, s) for c in CELLS for s in SEEDS]
    ):
        raise ValueError("targeted campaign source, runtime or roster differs")
    for cell, digest in plan["cells"].items():
        saved = sealed_read(root / "cells" / f"{cell}.json")
        if saved["artifact_sha256"] != digest or (
            saved["files"] != config()["public_files"][cell]
        ):
            raise ValueError("targeted saved cell differs")
    critic.checked_authorization(root, plan)
    return plan


def finish(root: Path) -> Path:
    """Freeze all six dispositions and export native models and audited receipts."""
    plan = verify(root)
    bundle = search.report(root, True)
    verify_controlled_bundle(root / "frozen_models.json", bundle)
    if sorted((r["benchmark_id"], r["repetition"]) for r in bundle["rows"]) != sorted(
        (c, s) for c in CELLS for s in SEEDS
    ):
        raise ValueError("frozen endpoint roster differs")
    audit = export_receipts(root, root / "endpoint-audit")
    if audit["counts"]["visits"] != 90:
        raise ValueError("expected ninety visit dispositions")
    rows = []
    for r in bundle["rows"]:
        if r["public_file_sha256"] != config()["public_files"][r["benchmark_id"]]:
            raise ValueError("endpoint public hashes differ")
        rows.append(
            {
                "cell": r["benchmark_id"],
                "seed": r["repetition"],
                "status": r["status"],
                "validation_nmse": r.get("validation_nmse"),
                "source_id": r["source_id"],
                "model_sha256": r.get("model_sha256"),
            }
        )
    search.write_csv(root / "validation_by_seed.csv", rows)
    final = root / "targeted-perturbation-results.zip"
    names = (
        "plan.json",
        "spec_variants.json",
        "ablation_switch_manifest.json",
        "critic_authorization.json",
        "frozen_models.json",
        "evaluation_freeze.json",
        "paired_ablation_results.csv",
        "search_visits.csv",
        "validation_by_seed.csv",
    )
    with TemporaryDirectory(dir=root) as tmp:
        folder = Path(tmp)
        (folder / "README.txt").write_text(
            "Two targeted 20B Full searches; three seeds and 15 visits per cell.\n"
            f"Plan: {plan['artifact_sha256']}\n"
            "Native models, context, fitted parameters and causal initializers are in "
            "frozen_models.json. Empty initials uses the candidate's analytic "
            "initializers. "
            "Retain missing-model dispositions. Numerical and mechanism intervention "
            "scoring belongs to the analyst, after this freeze. No test data opened.\n"
            "T1 retains the prior numeric data with a separately hashed "
            "exchange-required prompt. CSTR uses the supplied perturbed data "
            "and original public prompt.\n"
        )
        members = {n: root / n for n in names}
        members["README.txt"] = folder / "README.txt"
        for sub in ("endpoint-audit", "runtime", "submissions"):
            for p in sorted((root / sub).rglob("*")):
                if p.is_file() and p.suffix in {".json", ".csv", ".id"}:
                    members[str(p.relative_to(root))] = p
        for name in ("sacct.psv", "scheduler-cost.csv"):
            if (root / name).exists():
                members[name] = root / name
        recovery = root / "packaging_recovery.json"
        if recovery.is_file():
            members[recovery.name] = recovery
        (folder / "SHA256SUMS").write_text(
            "".join(
                f"{search.file_digest(p)}  {n}\n" for n, p in sorted(members.items())
            )
        )
        members["SHA256SUMS"] = folder / "SHA256SUMS"
        archive = folder / "export.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, path in members.items():
                z.write(path, name)
        with zipfile.ZipFile(archive) as z:
            if z.testzip():
                raise ValueError("export archive integrity failure")
        os.replace(archive, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "prepare", "verify", "finish"))
    for name in ("root", "public", "anchor", "gate", "archive", "original-t1"):
        parser.add_argument("--" + name, type=Path)
    a = parser.parse_args()
    required = {
        "stage": (a.archive, a.original_t1, a.public),
        "prepare": (a.root, a.public, a.anchor, a.gate),
        "verify": (a.root,),
        "finish": (a.root,),
    }[a.action]
    if not all(required):
        parser.error(f"missing paths for {a.action}")
    if a.action == "stage":
        result = stage(*required)
    elif a.action == "prepare":
        plan = prepare(*required)
        result = {"plan_sha256": plan["artifact_sha256"], "blocks": 6, "visits": 90}
    elif a.action == "verify":
        result = {"plan_sha256": verify(a.root)["artifact_sha256"]}
    else:
        result = {"archive": str(finish(a.root))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
