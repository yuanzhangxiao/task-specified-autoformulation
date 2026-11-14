"""Prepare independently freezable 120B Full cases in the user's requested order."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import iclr_aces_pilot as scheduler
from scripts import iclr_full_120b as comparison
from scripts import iclr_t1_perturbed_full as t1

CONFIG = comparison.search.io.REPO / "configs/iclr_full_120b_ordered_v1.json"
T1_PLAN = "eca069b46a5714df96230e381cd48e1003ce802df6ac765cce216fc32b2ca5b3"


def cases() -> list[dict]:
    """Return the committed order and public input fingerprints."""
    return json.loads(CONFIG.read_text())["cases"]


def case_named(name: str) -> dict:
    """Reject unknown case selectors before touching a campaign directory."""
    for case in cases():
        if case["key"] == name:
            return case
    raise ValueError(f"unknown original-case selector: {name}")


def prepare_case(
    base: Path, name: str, anchor: Path, t1_root: Path, gate: Path, public: Path
) -> dict:
    """Copy a verified 20B design; construct no model and open no test data."""
    case = case_named(name)
    original = comparison.checked_input(anchor / "plan.json", comparison.REFERENCE_PLAN)
    auth = comparison.critic.checked_authorization(
        anchor, {"artifact_sha256": comparison.REFERENCE_PLAN}
    )
    comparison.checked_input(gate / "plan.json", auth["calibration_plan_sha256"])
    if (
        content_hash(json.loads((gate / "summary.json").read_text()))
        != auth["summary_sha256"]
    ):
        raise ValueError("critic calibration differs from the original anchor")
    reference = anchor if case["reference"] == "anchors" else t1_root
    identity = comparison.REFERENCE_PLAN if reference == anchor else T1_PLAN
    if case["reference"] == "public":
        actual = {
            n: comparison.search.file_digest(public / case["cell"] / n)
            for n in case["public_files"]
        }
        if actual != case["public_files"]:
            raise ValueError("public files differ from the frozen original-case roster")
        reference = base / "public-designs" / name
        design = comparison.search.prepare(
            reference,
            public,
            [case["cell"]],
            ["full"],
            [0, 1, 2],
            15,
            True,
            True,
            (gate / "plan.json", gate / "summary.json"),
        )
        identity = design["artifact_sha256"]
    source = comparison.checked_input(reference / "plan.json", identity)
    if any(source[k] != original[k] for k in t1.CONFIG):
        raise ValueError("public Full configuration differs from the anchor")
    raw = comparison.checked_input(
        reference / "cells" / (case["cell"] + ".json"),
        source["cells"][case["cell"]],
    )
    if raw["files"] != case["public_files"]:
        raise ValueError("reference inputs differ from the frozen original-case roster")
    root = base / name
    plan = comparison.prepare(
        root, reference, gate, cells=[case["cell"]], reference_identity=identity
    )
    sealed_write(
        root / "ordered_case.json",
        {
            "protocol": "iclr-120b-original-case-1",
            "case": case,
            "roster_sha256": content_hash(json.loads(CONFIG.read_text())),
            "runner_sha256": comparison.search.file_digest(Path(__file__)),
            "plan_sha256": plan["artifact_sha256"],
            "reference": str(reference.resolve()),
            "reference_plan_sha256": identity,
            "has_matched_20b_run": case["reference"] != "public",
        },
    )
    return plan


def checked_case(base: Path, name: str) -> tuple[Path, dict, dict]:
    """Authenticate case identity and its independent three-seed roster."""
    root = base / name
    receipt = sealed_read(root / "ordered_case.json")
    plan = sealed_read(root / "plan.json")
    if (
        receipt["case"] != case_named(name)
        or receipt["roster_sha256"] != content_hash(json.loads(CONFIG.read_text()))
        or receipt["runner_sha256"] != comparison.search.file_digest(Path(__file__))
        or receipt["plan_sha256"] != plan["artifact_sha256"]
        or plan["seeds"] != [0, 1, 2]
        or plan["arms"] != ["full"]
        or [(b["cell"], b["seed"]) for b in plan["blocks"]]
        != [(receipt["case"]["cell"], s) for s in (0, 1, 2)]
    ):
        raise ValueError("ordered campaign identity or roster differs")
    return root, receipt, plan


def require_predecessors(base: Path, name: str) -> None:
    """Make earlier cases independently exportable before scheduling a later one."""
    for case in cases():
        if case["key"] == name:
            return
        prior, _, _ = checked_case(base, case["key"])
        source = prior / "frozen_models.json"
        if not source.exists():
            raise ValueError(f"finish and export {case['key']} before starting {name}")
        bundle = sealed_read(source)
        comparison.verify_controlled_bundle(source, bundle)
        identities = {
            (r["benchmark_id"], r["repetition"], r["method"]) for r in bundle["rows"]
        }
        if (
            len(bundle["rows"]) != 3
            or identities != {(case["cell"], s, "full") for s in (0, 1, 2)}
            or any(
                r["status"] not in {"ready", "no_retained_model"}
                for r in bundle["rows"]
            )
        ):
            raise ValueError("predecessor has an incomplete frozen roster")
    raise ValueError("unknown ordered case")


def finish_case(base: Path, name: str) -> Path:
    """Freeze this case and archive its models and public validation comparison."""
    root, receipt, _ = checked_case(base, name)
    comparison.finish(
        root,
        Path(receipt["reference"]),
        reference_identity=receipt["reference_plan_sha256"],
        allow_unmatched_reference=not receipt["has_matched_20b_run"],
    )
    filenames = [
        "ordered_case.json",
        "plan.json",
        "spec_variants.json",
        "ablation_switch_manifest.json",
        "critic_authorization.json",
        "frozen_models.json",
        "evaluation_freeze.json",
        "paired_ablation_results.csv",
        "search_visits.csv",
        "endpoint-audit",
    ]
    filenames += [
        n
        for n in ("paired_proposer_validation.csv", "validation_by_seed.csv")
        if (root / n).exists()
    ]
    target = root / "inspection.tar.gz"
    temporary = target.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for filename in filenames:
            archive.add(root / filename, arcname=filename)
    temporary.replace(target)
    return target


def main() -> None:
    """Prepare, launch, inspect or export one independent case."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run", "status", "finish"))
    p.add_argument("case", choices=[c["key"] for c in cases()] + ["all"])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--anchor", type=Path)
    p.add_argument("--t1", type=Path)
    p.add_argument("--gate", type=Path)
    p.add_argument("--public", type=Path)
    p.add_argument("--account")
    p.add_argument("--wave", default="wave1")
    a = p.parse_args()
    names = [c["key"] for c in cases()] if a.case == "all" else [a.case]
    if a.action in {"run", "finish"} and a.case == "all":
        p.error("run and finish require one named case")
    if a.action == "prepare" and not all((a.anchor, a.t1, a.gate, a.public)):
        p.error("prepare requires anchor, t1, gate and public directories")
    for name in names:
        if a.action == "prepare":
            plan = prepare_case(a.base, name, a.anchor, a.t1, a.gate, a.public)
            print(json.dumps({"case": name, "plan": plan["artifact_sha256"]}))
        elif a.action == "run":
            if not a.account:
                p.error("run requires --account")
            root, _, _ = checked_case(a.base, name)
            require_predecessors(a.base, name)
            print(
                scheduler.submit(root, a.wave, a.account, 1, confirm_critic_access=True)
            )
        elif a.action == "finish":
            print(finish_case(a.base, name))
        else:
            root, _, _ = checked_case(a.base, name)
            print(json.dumps({"case": name, "pending": scheduler.pending_ids(root)}))


if __name__ == "__main__":
    main()
