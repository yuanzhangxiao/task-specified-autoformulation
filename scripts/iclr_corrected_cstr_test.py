"""Score the frozen 16-arm historical CSTR rerun on corrected held-out trajectories."""

from __future__ import annotations

import argparse
import csv
import itertools
import tempfile
from pathlib import Path

import numpy as np

from autoformalism.benchmarks import reference_audit as audit
from autoformalism.benchmarks.phase_b_generation import phase_b_protocols
from autoformalism.benchmarks.phase_b_public import _write_split, phase_b_public_spec
from autoformalism.benchmarks.phase_c_release import SPEC_PATHS
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.data.models import SplitName
from autoformalism.fitting import FitConfig
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_campaign as campaign
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.postfreeze_evaluation import evaluate_subject_on_test
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts.iclr_cstr_correction import CSTR, EXPECTED
from scripts.iclr_submission_evidence import PLAN, digest

VERSION = "iclr-corrected-cstr-heldout-1"
GENERATOR_SPEC = "6a08872658c98ad7040b449a35cd6614b8f15030d04c1695a69e85baccc94a7a"


def check_roster(plan: dict, frozen: dict) -> None:
    """Require every final endpoint of the original four corrected CSTR blocks."""
    tasks = plan["tasks"]
    expected = set(
        itertools.product((0, 1), ("full", "brief_only"), (True, False), (True, False))
    )
    actual = {
        (t["seed"], t["arm"], t["critic"], t["scientific_verifier"]) for t in tasks
    }
    if (
        actual != expected
        or len(tasks) != 16
        or {t["block"] for t in tasks} != {16, 17, 18, 19}
        or any(t["cell"] != CSTR or not t["shared_processes"] for t in tasks)
    ):
        raise ValueError("require all 16 corrected CSTR campaign assignments")
    task_map = {t["task_id"]: t for t in tasks}
    if (
        {r["subject_id"] for r in frozen["roster"]} != {k + "_final" for k in task_map}
        or len(frozen["roster"]) != 16
        or len(task_map) != 16
        or frozen["plan_sha256"] != plan["artifact_sha256"]
        or frozen["further_search_permitted"] is not False
        or frozen["test_data_opened"] is not False
    ):
        raise ValueError("require complete final selection freeze before test access")
    for entry in frozen["roster"]:
        t = entry["task"]
        if (
            t != task_map[t["task_id"]]
            or entry["subject_id"] != t["task_id"] + "_final"
            or entry["endpoint"] != "final"
        ):
            raise ValueError("frozen final endpoint belongs to another assignment")


def intake(root, staging):
    """Verify source provenance without opening or generating held-out arrays."""
    plan = campaign.verify(root, execution=False)
    corrected = sealed_read(staging / "correction.json")
    if (
        corrected["protocol"] != "iclr-submitted-cstr-correction-1"
        or corrected["plan_sha256"] != PLAN
        or corrected["test_opened"] is not False
    ):
        raise ValueError("missing corrected development staging receipt")
    assets = plan["cells"][CSTR]["assets"]
    for cell, stored in plan["cells"].items():
        for name, sha in stored["assets"].items():
            if sha != corrected["public_files"][f"phase_b_v1/{cell}/{name}"]:
                raise ValueError("campaign does not match corrected development source")
    if any(assets[k] != v for k, v in EXPECTED.items()):
        raise ValueError("incorrect CSTR train, validation or prompt fingerprints")
    frozen = sealed_read(root / "evaluation_freeze.json")
    check_roster(plan, frozen)
    subjects_file = root / "subjects.jsonl"
    if digest(subjects_file) != frozen["subjects_sha256"]:
        raise ValueError("frozen subjects changed")
    subjects = {}
    for line in subjects_file.read_text().splitlines():
        s = FrozenEvaluationSubject.model_validate_json(line)
        if s.subject_id in subjects:
            raise ValueError("duplicate subject")
        subjects[s.subject_id] = s
    if (
        set(subjects)
        != {r["subject_id"] for r in frozen["roster"] if r["model_available"]}
        or len(subjects) != frozen["subject_count"]
    ):
        raise ValueError("available endpoints differ from the sealed subject file")
    for item in frozen["roster"]:
        if not item["model_available"]:
            continue
        s = subjects[item["subject_id"]]
        task = item["task"]
        if (
            s.benchmark_id != CSTR
            or s.tier != "easy"
            or s.repetition != task["seed"]
            or s.method != campaign.method_name(task, "final")
            or s.validation_context.lagged_targets
            or set(s.validation_context.targets) != {"T"}
            or set(s.validation_context.auxiliaries) != {"C", "Tj"}
            or set(s.validation_context.external_inputs) != {"Cf", "Tf", "Tjf"}
        ):
            raise ValueError("frozen model/cell channel or repetition mismatch")
    return plan, frozen, subjects, corrected["artifact_sha256"]


def split_rows(directory, data_root, split, identity):
    """Qualify the simulator and project its public channels."""
    if digest(data_root / SPEC_PATHS["cstr"]) != GENERATOR_SPEC:
        raise ValueError("reference system specification changed")
    spec = phase_b_public_spec(
        "cstr", "easy", "named", data_root=data_root, input_contract="legacy-events-1"
    )
    trajectories = []
    seals = {}
    for p in phase_b_protocols("cstr", input_contract="legacy-events-1"):
        if p.split != split:
            continue
        receipt, tr = audit.audit_one(
            directory / "numerics", p, "canonical", data_root, identity
        )
        if (
            not receipt["passed"]
            or tr.input_contract != "legacy-events-1"
            or tr.reference_generation_protocol != "reference-events-2"
        ):
            raise ValueError("corrected reference failed numerical qualification")
        trajectories.append(tr)
        seals[p.protocol_id] = receipt["record_sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        staged = Path(tmp) / f"{split}.csv"
        _write_split(staged, spec, tuple(trajectories))
        dest = directory / staged.name
        if dest.exists() and digest(dest) != digest(staged):
            raise ValueError("saved corrected CSV changed")
        if not dest.exists():
            staged.replace(dest)
    return dest, seals


def compare_development(original: Path, generated: Path):
    """Compare CSV schemas, IDs, sample times and all projected numerical channels."""
    a = np.genfromtxt(original, delimiter=",", names=True, dtype=None, encoding="utf-8")
    b = np.genfromtxt(
        generated, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    if (
        a.dtype.names != b.dtype.names
        or len(a) != len(b)
        or not np.array_equal(a["trajectory_id"], b["trajectory_id"])
        or not np.array_equal(a["t"], b["t"])
    ):
        raise ValueError("corrected development projection or grids differ")
    gaps = {
        k: float(np.max(abs(a[k] - b[k])) / max(1.0, np.max(abs(a[k]))))
        for k in a.dtype.names
        if k not in ("t", "trajectory_id")
    }
    if max(gaps.values()) > audit.AGREEMENT_LIMIT:
        raise ValueError("regenerated development trajectories differ from fitted ones")
    return gaps


def check_development(directory, data_root, corrected, identity):
    """Verify corrected public training and validation; this stage generates no test."""
    diagnostics = {}
    for name in ("train", "validation"):
        source = corrected / (name + ".csv")
        if digest(source) != EXPECTED[source.name]:
            raise ValueError("requires historical corrected CSV")
        path, seals = split_rows(directory, data_root, name, identity)
        diagnostics[name] = {
            "scaled_channel_gaps": compare_development(source, path),
            "regenerated_sha256": digest(path),
            "solver_checks": seals,
            "fitted_csv_sha256": digest(source),
        }
    return diagnostics


def run(root, staging, dest, data_root):
    """Run frozen subjects with the corrected test projection."""
    if any(
        dest.resolve().is_relative_to(p.resolve()) for p in (root, staging, data_root)
    ):
        raise ValueError("dedicate a separate evaluator-only output root")
    # Nothing in the private generator is opened before all selection checks pass.
    plan, freeze, subjects, source = intake(root, staging)
    public_dir = root / "public" / "phase_b_v1" / CSTR
    with public._lock(dest):
        key = sealed_write(
            dest / "plan.json",
            {
                "protocol": VERSION,
                "search_plan": plan["artifact_sha256"],
                "selection": freeze["artifact_sha256"],
                "corrected_staging": source,
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "evaluator_sha256": digest(Path(__file__)),
                "private_spec_sha256": GENERATOR_SPEC,
                "reference_protocol": "reference-events-2",
                "input_contract": "legacy-events-1",
                "subjects_sha256": freeze["subjects_sha256"],
                "development_assets": plan["cells"][CSTR]["assets"],
            },
        )["artifact_sha256"]
        checks = check_development(dest / "regenerated", data_root, public_dir, key)
        test_path, tests = split_rows(dest / "regenerated", data_root, "test", key)
        if len(tests) != 6:
            raise ValueError("incorrect corrected held-out roster")
        release = sealed_write(
            dest / "release.json",
            {
                "plan": key,
                "development": checks,
                "test_sha256": digest(test_path),
                "test_protocols": tests,
            },
        )
        registry = BenchmarkRegistry()
        loader = BenchmarkLoader(registry)
        spec = registry.get(CSTR)
        roles = loader._roles(spec, "easy")
        train = loader._load_tidy_split(
            spec, roles, SplitName.TRAIN, public_dir / "train.csv"
        )
        test = loader._load_tidy_split(spec, roles, SplitName.TEST, test_path)
        rows = []
        for entry in freeze["roster"]:
            sid = entry["subject_id"]
            task = entry["task"]
            result_path = dest / "results" / (sid + ".json")
            bind = {
                "plan": key,
                "release": release["artifact_sha256"],
                "subject_id": sid,
            }
            if not result_path.exists():
                started = result_path.with_suffix(".started.json")
                if not entry["model_available"]:
                    result = {"status": "model_unavailable"}
                elif started.exists():
                    if any(sealed_read(started).get(k) != v for k, v in bind.items()):
                        raise ValueError("bad reservation")
                    result = {"status": "interrupted"}
                else:
                    sealed_write(started, bind)
                    out = evaluate_subject_on_test(
                        subjects[sid],
                        training_split=train,
                        test_split=test,
                        fit_config=FitConfig(
                            integration_method="Radau",
                            allow_derivative_regression=False,
                            relative_tolerance=1e-7,
                            absolute_tolerance=1e-9,
                            maximum_wall_time_seconds=300,
                        ),
                    )
                    result = out.target_prediction.model_dump(mode="json")
                sealed_write(result_path, bind | {"prediction": result})
            rec = sealed_read(result_path)
            if any(rec[k] != v for k, v in bind.items()):
                raise ValueError("held-out result binding differs")
            rows.append(
                {
                    "subject_id": sid,
                    "seed": task["seed"],
                    "prompt": task["arm"],
                    "critic": task["critic"],
                    "verifier": task["scientific_verifier"],
                    **rec["prediction"],
                }
            )
        summary = sealed_write(
            dest / "corrected_cstr_test.json",
            {
                "protocol": VERSION,
                "plan": key,
                "dataset": release["artifact_sha256"],
                "source": "post-pruning final",
                "rows": rows,
                "parameter_refit": False,
            },
        )
        with (dest / "corrected_cstr_test.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
            )
            w.writeheader()
            w.writerows(rows)
        return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["development-check", "evaluate"])
    p.add_argument("--root", type=Path)
    p.add_argument("--staging", type=Path)
    p.add_argument("--corrected", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data_raw"))
    a = p.parse_args()
    if a.stage == "development-check":
        if not a.corrected:
            p.error("--corrected is required for development-check")
        with public._lock(a.output):
            found = check_development(
                a.output, a.data_root, a.corrected, "iclr-cstr-development-parity-1"
            )
            print(
                sealed_write(
                    a.output / "check.json",
                    {"test_trajectories_generated": False, "checks": found},
                )["artifact_sha256"]
            )
    else:
        if not a.root or not a.staging:
            p.error("evaluate requires --root and --staging")
        print(run(a.root, a.staging, a.output, a.data_root)["artifact_sha256"])
