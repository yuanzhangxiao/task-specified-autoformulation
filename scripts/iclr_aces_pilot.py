"""Live pilot receipts, shared-start checks, and bounded ACES submission waves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import iclr_controlled_search as study
from autoformalism.rebuttal import iclr_search_controls as controls
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.iclr_anchor_fidelity import verify_controlled_bundle
from scripts.submit_review_continuation import submit_job

PILOT_CELL = "acquisition_00"
PILOT_SEED = 97
PILOT_ARMS = ["full", "fixed_gamma"]


def verify_design(root: Path, original: Path, roster_path: Path) -> dict:
    """Bind a cluster-prepared plan to the packaged scientific design and its inputs."""
    plan, expected = (sealed_read(p) for p in (root / "plan.json", original))
    host_fields = {"artifact_sha256", "source_sha256", "runtime", "cells"}
    if {k: v for k, v in plan.items() if k not in host_fields} != {
        k: v for k, v in expected.items() if k not in host_fields
    } or set(plan["cells"]) != set(expected["cells"]):
        raise ValueError("new plan differs from frozen scientific design")
    roster = public._read(roster_path)
    for name, digest in plan["cells"].items():
        cell = sealed_read(root / "cells" / f"{name}.json")
        if (
            cell["artifact_sha256"] != digest
            or cell["files"] != roster["public_files"][name]
        ):
            raise ValueError("new plan differs from authenticated public inputs")
    return sealed_write(
        root / "launch_design.json",
        {
            "packaged_plan": expected["artifact_sha256"],
            "execution_plan": plan["artifact_sha256"],
            "execution_runtime": plan["runtime"],
            "packaged_runtime": expected["runtime"],
            "execution_source": plan["source_sha256"],
            "packaged_source": expected["source_sha256"],
            "host_cell_digests_changed": [
                k
                for k in sorted(plan["cells"])
                if plan["cells"][k] != expected["cells"][k]
            ],
            "public_files_verified": True,
            "identical_scientific_design": True,
        },
    )


def shared_start(root: Path, block: dict) -> dict:
    """Verify the two persisted round-zero records and their copied fit and critique."""
    tasks = {t["treatment"]: t for t in block["tasks"]}
    full, fixed = (io.round_path(root, tasks[a], 0) for a in PILOT_ARMS)
    original, copy = (sealed_read(d / "result.json") for d in (full, fixed))
    if (
        original["task"] != tasks["full"]
        or copy["task"] != tasks["fixed_gamma"]
        or copy["status"] != "shared_full_round_zero"
        or copy["shared_source_sha256"] != original["artifact_sha256"]
        or copy["cost"] != {}
    ):
        raise ValueError("incorrect shared-start lineage")
    selected = original["selected"]
    expected = (
        {**selected, "origin_task": tasks["fixed_gamma"]["task_id"]}
        if selected
        else None
    )
    if copy["selected"] != expected:
        raise ValueError(
            "shared fitted state, coefficients or retained proposal differ"
        )
    result = {
        "cell": block["cell"],
        "seed": block["seed"],
        "full_task": tasks["full"]["task_id"],
        "fixed_task": tasks["fixed_gamma"]["task_id"],
        "full_round_sha256": original["artifact_sha256"],
        "fixed_round_sha256": copy["artifact_sha256"],
        "available": bool(selected),
    }
    if selected:
        fit = public._read(full / "fit/result.json")
        if (
            fit != public._read(fixed / "fit/result.json")
            or fit["result"] != selected["fit"]
            or fit["sha256"] != public.content_sha256(selected["fit"])
        ):
            raise ValueError("shared fitter receipt differs")
        bundle = selected["bundle"]
        result.update(
            candidate_sha256=public.content_sha256(bundle["candidate"]),
            initialization_sha256=public.content_sha256(bundle["initialization"]),
            gamma_sha256=public.content_sha256(controls.gamma(bundle)),
            parameters_sha256=public.content_sha256(selected["fit"]["parameters"]),
            fit_sha256=fit["sha256"],
        )
        if tasks["full"]["critic"]:
            a, b = (sealed_read(d / "critic.json") for d in (full, fixed))
            if a != b or a["proposal_sha256"] != original["proposal_sha256"]:
                raise ValueError("shared initial critic receipt differs")
            result["critic_sha256"] = a["artifact_sha256"]
    return result


def pair_receipts(root: Path) -> dict:
    """Bind actual shared starts to the complete exported roster, including failures."""
    models = root / "frozen_models.json"
    frozen = sealed_read(models)
    verify_controlled_bundle(models, frozen, allow_development_pilot=True)
    plan = sealed_read(root / "plan.json")
    return sealed_write(
        root / "shared_initial_receipts.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "frozen_models_sha256": frozen["artifact_sha256"],
            "pairs": [shared_start(root, b) for b in plan["blocks"]],
        },
    )


def checkpoint_digest(root: Path) -> str:
    """Hash the scientific and call checkpoints before a no-op restart."""
    parts = {}
    for directory in ("results", "judge_cache", "pruning"):
        for f in sorted((root / directory).rglob("*.json")):
            parts[f.relative_to(root).as_posix()] = study.file_digest(f)
    return public.content_sha256(parts)


def pending_ids(root: Path) -> list[int]:
    """Recompute coverage between waves and return blocks with unfinished work."""
    study.report(root, False)
    plan = sealed_read(root / "plan.json")
    with (root / "paired_ablation_results.csv").open() as stream:
        pending = {
            r["source_id"] for r in csv.DictReader(stream) if r["status"] == "pending"
        }
    return [
        b["block"]
        for b in plan["blocks"]
        if any(t["task_id"] in pending for t in b["tasks"])
    ]


def checked_pilot_plan(root: Path) -> dict:
    """Require the declared engineering pilot and its calibrated critic."""
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] != study.PROTOCOL
        or not plan.get("development_pilot")
        or plan["arms"] != PILOT_ARMS
        or plan["visits"] != 2
        or not plan["critic_enabled"]
        or not plan["pruning_enabled"]
        or [(b["cell"], b["seed"]) for b in plan["blocks"]]
        != [(PILOT_CELL, PILOT_SEED)]
    ):
        raise ValueError("not the declared two-visit development pilot")
    critic.checked_authorization(root, plan)
    return plan


def check_cached_pilot(root: Path) -> None:
    """Verify cached model calls and test writes in the remaining fit directories."""
    plan = checked_pilot_plan(root)
    if (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("controlled-search implementation or runtime changed")
    with io.execution_lease(root, exclusive=True):
        io.require_open(root)
        destinations = {root, root / "logs"}
        for task in plan["blocks"][0]["tasks"]:
            pruning = root / "pruning" / task["task_id"]
            destinations.update((pruning, pruning / "control", pruning / "pruned"))
            for visit in range(plan["visits"]):
                if io.read_round(root, task, visit) is not None:
                    continue
                directory = io.round_path(root, task, visit)
                proposal_path = directory / "proposal.json"
                review_path = directory / "critic.json"
                if not proposal_path.is_file() or not review_path.is_file():
                    raise ValueError(
                        f"missing saved proposal or critic review at {directory}"
                    )
                proposal, review = sealed_read(proposal_path), sealed_read(review_path)
                if (
                    proposal["task"] != task
                    or proposal["round"] != visit
                    or review["task"] != task
                    or review["round"] != visit
                    or review["identity"] != plan["artifact_sha256"]
                    or review["proposal_sha256"] != proposal["artifact_sha256"]
                ):
                    raise ValueError(f"cached visit lineage differs at {directory}")
                destinations.update((directory, directory / "fit"))
        for directory in sorted(destinations):
            directory.mkdir(parents=True, exist_ok=True)
            # Sibling directories on ACES can have different Lustre metadata targets.
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".resume-write-"
            ) as probe:
                probe.write(b"checkpoint\n")
                probe.flush()
                os.fsync(probe.fileno())


def run_cached_pilot(root: Path, seconds: int) -> dict:
    """Resume cached proposals on CPUs and use Jetstream for new final reviews."""
    check_cached_pilot(root)
    if not os.environ.get("AF_JETSTREAM_API_KEY"):
        raise ValueError("supply the critic credential before any final review starts")

    def no_proposer(*args):
        raise RuntimeError("CPU resume attempted a new proposer call")

    return study.run_block(
        root,
        0,
        "http://127.0.0.1:1",
        seconds,
        transport=no_proposer,
        token_transport=no_proposer,
    )


def finish(root: Path) -> dict:
    """Check idempotent resume, then freeze and audit the engineering pilot."""
    plan = checked_pilot_plan(root)
    resume_path = root / "pilot_resume.json"
    if not resume_path.exists():
        if pending_ids(root):
            raise ValueError("pilot still has pending work; resume the same root")
        before = checkpoint_digest(root)

        def no_request(*args):
            raise RuntimeError("completed pilot attempted a new provider call")

        # A complete run already has its critic/pruning receipts. The CLI for this
        # audit runs without credentials; an accidentally reopened stage cannot spend.
        study.run_block(
            root,
            0,
            "http://127.0.0.1:1",
            600,
            transport=no_request,
            token_transport=no_request,
        )
        after = checkpoint_digest(root)
        if before != after:
            raise ValueError(
                "completed pilot changed scientific or call checkpoints on resume"
            )
        sealed_write(
            resume_path,
            {"plan_sha256": plan["artifact_sha256"], "checkpoints_sha256": before},
        )
    resume = sealed_read(resume_path)
    if resume["plan_sha256"] != plan["artifact_sha256"] or resume[
        "checkpoints_sha256"
    ] != checkpoint_digest(root):
        raise ValueError("pilot checkpoints changed after resume verification")
    frozen = study.report(root, True)
    pairs = pair_receipts(root)
    starts = pairs["pairs"][0]
    full, fixed = plan["blocks"][0]["tasks"]
    services = {}
    for file in (root / "runtime").glob("service-*.json"):
        service = public._read(file)
        if (
            service["plan_sha256"] != plan["artifact_sha256"]
            or service["image_sha256"] != plan["serving_image_sha256"]
            or service["model"] != plan["model_settings"]["model"]
            or service["revision"] != plan["model_settings"]["model_revision"]
            or service["block"] != 0
            or urlparse(service["endpoint"]).hostname != "127.0.0.1"
        ):
            raise ValueError("pilot vLLM service differs from the scheduled plan")
        services[service["endpoint"]] = study.file_digest(file)
    calls = [public._read(f) for f in (root / "results").glob("*/round_*/calls/*.json")]
    live = [
        c
        for c in calls
        if c["status"] == "responded"
        and c["endpoint"] in services
        and c["request"]["settings"]["model_revision"]
        == plan["model_settings"]["model_revision"]
    ]
    reviews = [sealed_read(f) for f in (root / "results").glob("*/round_*/critic.json")]
    wire = critic.cost(root)
    available_reviews = [
        r
        for r in reviews
        if r.get("review", {}).get("status") == "reviewed"
        and not r.get("reused")
        and any(
            (root / "judge_cache" / r["request_sha256"] / "wire").glob(
                "call-*/result.json"
            )
        )
    ]
    fixed_call_dir = io.round_path(root, fixed, 1) / "calls"
    sent_fixed = any(
        "frozen_dynamic_representation"
        in json.dumps(public._read(f)["request"]["body"])
        for f in fixed_call_dir.glob("*.json")
    )
    prunes = {
        t["treatment"]: sealed_read(root / "pruning" / t["task_id"] / "result.json")[
            "status"
        ]
        for t in (full, fixed)
    }
    fixed_row = next(
        (row for row in frozen["rows"] if row.get("source_id") == fixed["task_id"]),
        {},
    )
    fixed_definitions_preserved = False
    if starts["available"] and fixed_row.get("status") == "ready":
        initial = io.read_round(root, fixed, 0)["selected"]
        final_visit = io.read_round(root, fixed, plan["visits"] - 1)["selected"]
        before_gamma = controls.gamma(initial["bundle"])
        exported_gamma = controls.gamma(
            {
                "candidate": fixed_row["candidate"],
                "initialization": {"context": fixed_row["context"]},
            }
        )
        fixed_definitions_preserved = controls.gamma(
            final_visit["bundle"]
        ) == before_gamma and all(
            exported_gamma[k] == before_gamma[k] for k in ("states", "direct")
        )
    checks = {
        "loopback_proposer_responded": len(live) > 0,
        "shared_fitted_start": starts["available"],
        "complete_live_critic_review": bool(available_reviews)
        and wire["physical_requests_started"] > 0,
        "fixed_revision_constraint_sent": sent_fixed,
        "fixed_representation_preserved": fixed_definitions_preserved,
        "pruning_dispositions": all(
            x != "skipped_no_finite_parent" for x in prunes.values()
        ),
    }
    return sealed_write(
        root / "pilot_receipt.json",
        {
            "protocol": "iclr-live-development-pilot-1",
            "plan_sha256": plan["artifact_sha256"],
            "source_sha256": plan["source_sha256"],
            "runtime": plan["runtime"],
            "serving_image_sha256": plan["serving_image_sha256"],
            "checks": checks,
            "qualified": all(checks.values()),
            "pruning": prunes,
            "proposer_responses": len(live),
            "service_receipts": services,
            "proposer_requests": sum(
                r["cost"].get("proposal_calls", 0) for r in frozen["rows"]
            ),
            "observed_proposer_tokens": sum(
                r["cost"].get("observed_proposal_tokens", 0) for r in frozen["rows"]
            ),
            "unknown_proposer_token_counts": sum(
                r["cost"].get("unknown_proposal_token_counts", 0)
                for r in frozen["rows"]
            ),
            "critic_transport": wire,
            "resume_sha256": resume["artifact_sha256"],
            "pairs_sha256": pairs["artifact_sha256"],
            "models_sha256": frozen["artifact_sha256"],
            "private_reference_opened": False,
            "test_data_opened": False,
        },
    )


def submit(
    root: Path, wave: str, account: str, concurrency: int, *, cached_pilot: bool = False
) -> str:
    """Submit unfinished blocks once per wave and prevent overlapping waves."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", wave) or concurrency not in (1, 2):
        raise ValueError("use a named wave and concurrency one or two")
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != study.PROTOCOL:
        raise ValueError("expected a controlled campaign")
    critic.checked_authorization(root, plan)
    if not os.environ.get("AF_JETSTREAM_API_KEY"):
        raise ValueError("set the critic credential in the submitting shell")
    submissions = root / "submissions"
    with public._lock(submissions):
        folder = submissions / wave
        jobfile = folder / "array.id"
        if jobfile.exists() and jobfile.read_text().strip().isdecimal():
            return jobfile.read_text().strip()
        for intent in submissions.glob("*/array.intent.json"):
            ident = intent.with_name("array.id")
            if not ident.exists() or not ident.read_text().strip().isdecimal():
                raise ValueError(
                    f"uncertain Slurm reply; reconcile {intent.parent} with sacct"
                )
        prior = [p.read_text().strip() for p in submissions.glob("*/array.id")]
        if prior:
            ids = subprocess.check_output(
                ["squeue", "-h", "-u", os.environ["USER"], "-o", "%F"], text=True
            )
            if set(ids.split()).intersection(prior):
                raise ValueError("a previous wave is active; wait for it to exit")
        io.require_open(root)
        if cached_pilot:
            if concurrency != 1:
                raise ValueError("cached pilot requires concurrency one")
            check_cached_pilot(root)
        indices = pending_ids(root)
        if not indices:
            return "no pending blocks"
        folder.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        array = ",".join(map(str, indices)) + f"%{concurrency}"
        os.environ["AF_SEARCH_ROOT"] = str(root.resolve())
        worker = io.REPO / "scripts/hpc/run_iclr_controlled_search_aces.sh"
        partition = "gpu"
        if cached_pilot:
            helper = Path(__file__).resolve()
            os.environ["AF_PILOT_RESUME_CLI"] = str(helper)
            worker = helper.parent / "hpc/run_iclr_cached_pilot_aces.sh"
            partition = "cpu"
        return submit_job(
            folder,
            "array",
            [
                f"--account={account}",
                f"--partition={partition}",
                f"--array={array}",
                "--export=ALL",
                f"--output={root}/logs/{wave}-%A_%a.out",
                f"--error={root}/logs/{wave}-%A_%a.err",
            ],
            worker,
            "run",
            0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser.add_subparsers(dest="stage", required=True)
    for cmd in (
        "prepare",
        "finish",
        "check-design",
        "submit",
        "check-cached",
        "run-cached",
    ):
        sub = p.add_parser(cmd)
        sub.add_argument("--root", type=Path, required=True)
        if cmd == "prepare":
            sub.add_argument("--public", type=Path, required=True)
        if cmd == "check-design":
            sub.add_argument("--original", type=Path, required=True)
            sub.add_argument("--roster", type=Path, required=True)
        if cmd == "submit":
            sub.add_argument("--cached-pilot", action="store_true")
            sub.add_argument("--wave", required=True)
            sub.add_argument("--account", required=True)
            sub.add_argument("--concurrency", type=int, default=2)
        if cmd == "run-cached":
            sub.add_argument("--seconds", type=int, required=True)
    a = parser.parse_args()
    if a.stage == "prepare":
        result = study.prepare(
            a.root,
            a.public,
            [PILOT_CELL],
            PILOT_ARMS,
            [PILOT_SEED],
            2,
            True,
            True,
            development_pilot=True,
        )
        print(result["artifact_sha256"])
    elif a.stage == "submit":
        print(
            submit(
                a.root, a.wave, a.account, a.concurrency, cached_pilot=a.cached_pilot
            )
        )
    elif a.stage == "check-cached":
        check_cached_pilot(a.root)
        print("Cached pilot verified. Remaining checkpoint directories are writable.")
    elif a.stage in ("run-cached", "finish"):
        if a.stage == "run-cached":
            progress = run_cached_pilot(a.root, a.seconds)
            print(json.dumps(progress))
            if progress["status"] != "complete":
                return
            os.environ.pop("AF_JETSTREAM_API_KEY", None)
        result = finish(a.root)
        print(json.dumps(result, indent=2))
        if not result["qualified"]:
            raise SystemExit(
                "pilot finished with incomplete operational coverage; inspect checks"
            )
    else:
        print(verify_design(a.root, a.original, a.roster)["artifact_sha256"])


if __name__ == "__main__":
    main()
