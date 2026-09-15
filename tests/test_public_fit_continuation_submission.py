"""The one-window CPU launcher preserves parent inputs and submission identity."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class Launcher:
    repo: Path
    root: Path
    parent: Path
    selection: Path
    bins: Path
    log: Path
    env: dict[str, str]

    def call(self, script: str, **updates: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.repo / "scripts/hpc" / script)],
            env={**self.env, **updates},
            capture_output=True,
            text=True,
        )

    def submit(self, **updates: str) -> subprocess.CompletedProcess:
        return self.call("submit_public_fit_continuation_aces.sh", **updates)

    def worker(self, **updates: str) -> subprocess.CompletedProcess:
        return self.call("run_public_fit_continuation_aces.sh", **updates)

    def python(self, body: str) -> str:
        path = self.bins / "fixture-python"
        path.write_text(
            f"#!{sys.executable}\n"
            "import subprocess, sys\n"
            "if sys.argv[1:2] == ['-']:\n"
            "    result = subprocess.run([sys.executable, *sys.argv[1:]])\n"
            "    raise SystemExit(result.returncode)\n"
            f"{body}\n"
        )
        path.chmod(0o755)
        return str(path)


@pytest.fixture
def launcher(tmp_path):
    if not all(shutil.which(c) for c in ("bash", "jq", "sha256sum")):
        pytest.skip("shell tools unavailable")
    repo = Path(__file__).resolve().parents[1]
    bins, parent = tmp_path / "bin", tmp_path / "parent experiment" / "fit"
    bins.mkdir()
    parent.mkdir(parents=True)
    for name in ("freeze.json", "result.json", "backend_result.json"):
        (parent / name).write_text(json.dumps({"fixture": name}))
    selection = tmp_path / "selection.json"
    selection.write_text('{"fixture": "one-window-pilot"}')
    root, log = tmp_path / "output root", tmp_path / "scheduler.json"
    scripts = {
        "module": "#!/bin/sh\nexit 0\n",
        "git": (
            '#!/bin/sh\ncase "$*" in *status*) '
            '[ "${DIRTY:-0}" = 0 ] || echo " M src/changed.py"; exit 0 ;; '
            '*) echo "${REVISION:-abc123}" ;; esac\n'
        ),
        "sbatch": f"""#!{sys.executable}
import json, os, pathlib, sys
path = pathlib.Path({str(log)!r})
root = pathlib.Path(os.environ['AF_OUTPUT_ROOT'])
identity = json.loads((root / 'submission-intent/identity.json').read_text())
calls = json.loads(path.read_text()) if path.exists() else []
calls.append({{'args': sys.argv[1:], 'identity': identity,
              'final_manifest_present': (root / 'submission_manifest.json').exists()}})
path.write_text(json.dumps(calls))
print(os.environ.get('JOB_REPLY', '501;aces'))
sys.exit(int(os.environ.get('FAIL_SUBMIT', '0')))
""",
    }
    for name, body in scripts.items():
        path = bins / name
        path.write_text(body)
        path.chmod(0o755)
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("AF_")},
        "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(repo),
        "AF_OUTPUT_ROOT": str(root),
        "AF_PARENT_FIT": str(parent),
        "AF_SELECTION": str(selection),
        "AF_PYTHON": sys.executable,
    }
    return Launcher(repo, root, parent, selection, bins, log, env)


def test_one_cpu_submission_binds_parent_and_reuses_manifest(launcher):
    before = {p.name: p.read_bytes() for p in launcher.parent.iterdir()}
    first = launcher.submit()
    assert first.returncode == 0, first.stderr
    calls = json.loads(launcher.log.read_text())
    assert len(calls) == 1
    args = calls[0]["args"]
    assert {"--partition=cpu", "--cpus-per-task=1", "--mem=16G"} <= set(args)
    assert "--time=00:30:00" in args
    assert not any("--gres" in arg for arg in args)
    assert not calls[0]["final_manifest_present"]
    manifest = json.loads(first.stdout)
    assert manifest["job_id"] == "501"
    assert manifest["identity"] == calls[0]["identity"]
    assert all(
        len(manifest["identity"][key]) == 64
        for key in (
            "parent_freeze_sha256",
            "parent_result_sha256",
            "parent_backend_sha256",
            "selection_sha256",
        )
    )
    assert launcher.submit().stdout == first.stdout
    assert json.loads(launcher.log.read_text()) == calls
    assert {p.name: p.read_bytes() for p in launcher.parent.iterdir()} == before


@pytest.mark.parametrize("updates", [{"FAIL_SUBMIT": "1"}, {"JOB_REPLY": "invalid"}])
def test_uncertain_submission_cannot_duplicate(launcher, updates):
    assert launcher.submit(**updates).returncode != 0
    assert (launcher.root / "submission-intent/identity.json").exists()
    assert launcher.submit().returncode != 0
    assert len(json.loads(launcher.log.read_text())) == 1


@pytest.mark.parametrize("change", ["commit", "selection", "parent"])
def test_changed_identity_cannot_reuse_submission(launcher, change):
    assert launcher.submit().returncode == 0
    updates = {}
    if change == "commit":
        updates["REVISION"] = "different"
    elif change == "selection":
        launcher.selection.write_text("{}")
    else:
        (launcher.parent / "result.json").write_text("{}")
    result = launcher.submit(**updates)
    assert result.returncode != 0
    assert "identity changed" in result.stderr
    assert len(json.loads(launcher.log.read_text())) == 1


@pytest.mark.parametrize(
    "location", ["inside", "ancestor", "sibling", "symlink", "checkout"]
)
def test_output_cannot_overlap_source_paths(launcher, location):
    if location == "inside":
        output = launcher.parent / "child"
    elif location == "ancestor":
        output = launcher.parent.parent
    elif location == "sibling":
        output = launcher.parent.parent / "new-output"
    elif location == "checkout":
        output = launcher.repo / "continuation-output"
    else:
        alias = launcher.root.parent / "parent-alias"
        alias.symlink_to(launcher.parent, target_is_directory=True)
        output = alias / "child"
    result = launcher.submit(AF_OUTPUT_ROOT=str(output))
    assert result.returncode != 0
    assert "Output must be separate from the parent experiment" in result.stderr
    assert not launcher.log.exists()


@pytest.mark.parametrize("alias", [False, True])
def test_worker_rejects_old_experiment_output_even_with_matching_intent(
    launcher, alias
):
    assert launcher.submit().returncode == 0
    experiment = launcher.parent.parent
    if alias:
        experiment = launcher.root.parent / "experiment-alias"
        experiment.symlink_to(launcher.parent.parent, target_is_directory=True)
    output = experiment / "new-output"
    intent = output / "submission-intent"
    intent.mkdir(parents=True)
    identity = json.loads(
        (launcher.root / "submission-intent/identity.json").read_text()
    )
    identity["output_root"] = str(output)
    (intent / "identity.json").write_text(json.dumps(identity))
    before = {
        str(p.relative_to(launcher.parent.parent)): p.read_bytes()
        for p in launcher.parent.parent.rglob("*")
        if p.is_file()
    }
    result = launcher.worker(AF_OUTPUT_ROOT=str(output))
    assert result.returncode == 2
    assert "Output must be separate from the parent experiment" in result.stderr
    assert not (output / "runtime").exists()
    assert not (output / "continuation").exists()
    assert before == {
        str(p.relative_to(launcher.parent.parent)): p.read_bytes()
        for p in launcher.parent.parent.rglob("*")
        if p.is_file()
    }


def test_dirty_checkout_or_missing_parent_rejected_before_submission(launcher):
    assert launcher.submit(DIRTY="1").returncode != 0
    (launcher.parent / "backend_result.json").unlink()
    assert launcher.submit().returncode != 0
    assert not launcher.log.exists()


def test_existing_continuation_is_not_resubmitted_without_manifest(launcher):
    (launcher.root / "continuation").mkdir(parents=True)
    result = launcher.submit()
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert not launcher.log.exists()


def test_worker_surfaces_preflight_failure_without_touching_parent(launcher):
    assert launcher.submit().returncode == 0
    python = launcher.python("print('fixture dependency failure')\nraise SystemExit(3)")
    result = launcher.worker(AF_PYTHON=python)
    assert result.returncode == 3
    assert "fixture dependency failure" in result.stderr
    assert not (launcher.root / "continuation").exists()


@pytest.mark.parametrize(
    "change",
    ["commit", "dirty", "selection", "freeze", "result", "backend_result", "path"],
)
def test_worker_rejects_queued_drift_before_preflight(launcher, change):
    assert launcher.submit().returncode == 0
    updates = {}
    if change == "commit":
        updates["REVISION"] = "different"
    elif change == "dirty":
        updates["DIRTY"] = "1"
    elif change == "selection":
        launcher.selection.write_text("{}")
    elif change == "path":
        updates["AF_PARENT_FIT"] = str(launcher.parent.parent)
    else:
        (launcher.parent / f"{change}.json").write_text("{}")
    result = launcher.worker(**updates)
    assert result.returncode == 2
    assert "changed after submission" in result.stderr
    assert not (launcher.root / "runtime").exists()


def test_fast_worker_needs_only_prepublished_identity(launcher):
    assert launcher.submit().returncode == 0
    (launcher.root / "submission_manifest.json").unlink()
    python = launcher.python("print('preflight reached')\nraise SystemExit(3)")
    result = launcher.worker(AF_PYTHON=python)
    assert result.returncode == 3
    assert "preflight reached" in result.stderr


def test_worker_rejects_manifest_disagreement(launcher):
    assert launcher.submit().returncode == 0
    path = launcher.root / "submission_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["commit"] = "different"
    path.write_text(json.dumps(manifest))
    result = launcher.worker()
    assert result.returncode == 2
    assert "manifest identity differs" in result.stderr


@pytest.mark.parametrize("drift", [False, True])
def test_worker_sequence_and_second_guard(launcher, drift):
    assert launcher.submit().returncode == 0
    calls = launcher.root / "python-calls.json"
    python = launcher.python(f"""
import json, os, pathlib, sys
args = sys.argv[1:]
path = pathlib.Path({str(calls)!r})
calls = json.loads(path.read_text()) if path.exists() else []
calls.append({{'args': args, 'tmpdir': os.environ.get('TMPDIR')}})
path.write_text(json.dumps(calls))
if args[:2] == ['-m', 'pytest']:
    print('fixture preflight passed')
elif args[0] == '-c':
    print('{{"fixture": "environment"}}')
elif args[0] == 'scripts/smoke_public_fit_continuation.py':
    if {drift!r}:
        pathlib.Path(os.environ['AF_SELECTION']).write_text('{{}}')
    print('{{"fixture": "smoke"}}')
elif args[0] == 'scripts/run_public_fit_continuation.py':
    command = args[1]
    directory = pathlib.Path(args[args.index('--output') + 1])
    if command == 'prepare':
        directory.mkdir()
    print(json.dumps({{'fixture': command}}))
else:
    raise SystemExit('Unexpected fixture call')
""")
    before = {p.name: p.read_bytes() for p in launcher.parent.iterdir()}
    result = launcher.worker(AF_PYTHON=python)
    records = json.loads(calls.read_text())
    core = [
        r["args"]
        for r in records
        if r["args"][0].endswith("run_public_fit_continuation.py")
    ]
    assert all(r["tmpdir"] == str(launcher.root / "tmp") for r in records)
    assert "no:cacheprovider" in records[0]["args"]
    if drift:
        assert result.returncode == 2
        assert "changed after submission" in result.stderr
        assert core == []
        assert not (launcher.root / "continuation").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert [r[1] for r in core] == ["prepare", "inspect", "run", "report"]
        assert core[0][core[0].index("--parent") + 1] == str(launcher.parent)
        assert all(
            r[r.index("--output") + 1] == str(launcher.root / "continuation")
            for r in core
        )
        assert json.loads((launcher.root / "summary.json").read_text()) == {
            "fixture": "report"
        }
    assert {p.name: p.read_bytes() for p in launcher.parent.iterdir()} == before
