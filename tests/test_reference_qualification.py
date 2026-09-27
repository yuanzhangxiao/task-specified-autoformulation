"""Release integrity, public-information boundaries and bounded diagnostic resume."""

import json
from pathlib import Path

import pytest

from autoformalism.benchmarks import audited_release as release
from autoformalism.benchmarks import reference_audit as audit
from autoformalism.benchmarks import reference_qualification as q
from autoformalism.benchmarks.phase_b_generation import phase_b_protocols
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitResult, PublicSplit


@pytest.fixture
def audited(tmp_path, monkeypatch):
    data = Path("data_raw")
    if not (data / q.SPEC_PATH).exists():
        pytest.skip("private evaluator fixture not installed")
    suite = json.loads(Path("configs/benchmarks/phase_b_suite_v1.json").read_text())
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite))
    full_suite = release.load_suite_spec(suite_path)
    # Exercise the real publication path on a bounded CSTR test fixture.
    # Production suite validation remains unchanged (all three families required).
    subset = full_suite.model_copy(
        update={"families": tuple(f for f in full_suite.families if f.family == "cstr")}
    )
    monkeypatch.setattr(release, "load_suite_spec", lambda _: subset)
    base = phase_b_protocols("cstr", input_contract="continuous-rates-1")[0]
    protocols = tuple(
        base.model_copy(
            update={
                "protocol_id": f"{split}_unit",
                "split": split,
                "duration": 2,
                "dt": 0.1,
                "input_dt": 0.1,
                "specification": {"kind": "constant", "value": [0.1, 0.1, -0.1]},
            }
        )
        for split in ("train", "validation", "test")
    )
    monkeypatch.setattr(release, "phase_b_protocols", lambda *a, **k: protocols)
    root = tmp_path / "audit"
    plan = {
        "protocol": "phase-b-reference-integrity-audit-1",
        "input_contract": "continuous-rates-1",
        "include_test_protocols": True,
        "suite_sha256": audit._digest(suite_path),
        "private_spec_sha256": {str(q.SPEC_PATH): audit._digest(data / q.SPEC_PATH)},
    }
    root.mkdir()
    public._write(root / "plan.json", plan)
    for p in protocols:
        audit.audit_one(
            root / "private/cstr-controlled_reactor_mechanism-canonical",
            p,
            "canonical",
            data,
            audit._hash(plan),
        )
    public._write(
        root / "summary.json",
        {
            "identity": audit._hash(plan),
            "numerical_checks_passed": True,
            "numerical_protocols": 3,
        },
    )
    return root, data, suite_path, tmp_path / "public"


def test_publish_resume_checks_prompts_arrays_and_preserves_audit(audited):
    root, data, suite, output = audited
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    first = release.publish_audited_release(*audited)
    assert first["released_cells"] == 4
    assert release.publish_audited_release(*audited) == first
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    prompt = next(output.rglob("proposer_prompt.txt"))
    prompt.write_text(prompt.read_text() + "changed")
    with pytest.raises(ValueError, match="published cell differs"):
        release.publish_audited_release(root, data, suite, output)


@pytest.mark.parametrize("defect", ["missing", "array", "failed_audit", "legacy"])
def test_unqualified_cache_is_not_regenerated(audited, monkeypatch, defect):
    root, *_ = audited
    if defect == "missing":
        next(root.rglob("*.npz")).unlink()
    elif defect == "array":
        next(root.rglob("*.npz")).write_bytes(b"broken")
    else:
        path = root / ("summary.json" if defect == "failed_audit" else "plan.json")
        value = public._read(path)
        value[
            "numerical_checks_passed" if defect == "failed_audit" else "input_contract"
        ] = False if defect == "failed_audit" else "legacy-events-1"
        public._write(path, value)
    monkeypatch.setattr(
        audit, "simulate_phase_b", lambda *a, **k: pytest.fail("regeneration")
    )
    with pytest.raises(ValueError):
        release.publish_audited_release(*audited)


def test_public_preparation_never_reads_test_and_drops_private_labels(
    audited, tmp_path, monkeypatch
):
    release.publish_audited_release(*audited)
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert self.name != "test.csv", "fitting preparation opened test data"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    root = tmp_path / "diagnostic"
    q.prepare(root, audited[-1], audited[1])
    assert q.prepare(root, audited[-1], audited[1])["live_llm_calls"] == 0
    frozen = public._read(root / "cstr_hard_generic/fit/freeze.json")
    rows = frozen["training"]["rows"]
    assert all(set(r["targets"]) == {"T"} and not r["auxiliaries"] for r in rows)
    assert all("derivatives" not in r for r in rows)
    assert {
        i["state"]: i["expression"]
        for i in frozen["lowered_candidate"]["initial_conditions"]
    } == {
        "C": "init_C_a + init_C_b * (T - 350)",
        "T": "T",
        "Tj": "init_Tj_a + init_Tj_b * (T - 350)",
    }
    assert q.report(root)["status"] == "incomplete"
    easy = public._read(root / "cstr_easy_generic/fit/freeze.json")
    assert len(easy["lowered_candidate"]["states"]) == 3
    assert {
        i["state"]: i["expression"]
        for i in easy["lowered_candidate"]["initial_conditions"]
    } == {
        "c_internal": "C",
        "T": "T",
        "j_internal": "Tj",
    }


@pytest.mark.parametrize("index", [1, 2])
def test_reference_transcription_and_fit_resume(audited, tmp_path, monkeypatch, index):
    release.publish_audited_release(*audited)
    root = tmp_path / "diagnostic"
    q.prepare(root, audited[-1], audited[1])
    # Real two-solver free replay, fake expensive optimizer only.
    calls = []

    def fit(directory):
        calls.append(directory)
        frozen, request, _, _ = public._load(directory)
        truth = release.read_seal(directory.parent / "reference_parameters.json")
        return PublicFitResult(
            status="complete",
            profile=request.profile,
            identity=frozen["identity"],
            request_sha256=public.content_sha256(request),
            lowered_candidate_sha256=public.content_sha256(frozen["lowered_candidate"]),
            initialization_plan_sha256=public.content_sha256(
                request.initialization_plan
            ),
            training_content_sha256=public.content_sha256(frozen["training"]),
            validation_content_sha256=public.content_sha256(frozen["validation"]),
            source=request.source,
            parameters=truth,
            message="test backend",
        )

    monkeypatch.setattr(public, "execute_fit", fit)
    result = q.run_task(root, index)
    assert result["reference"]["metrics"]["val"]["nmse"] < 1e-8
    assert result["fitted_replay"]["complete"]
    assert q.run_task(root, index) == result
    assert len(calls) == 2  # public.execute_fit owns optimizer resume, not this wrapper
    with pytest.raises(ValueError, match="task index"):
        q.run_task(root, -1)
    frozen, request, train, val = public._load(root / "cstr_hard_generic/fit")
    truth = release.read_seal(root / "cstr_hard_generic/reference_parameters.json")
    wrong = q.replay(request, {**truth, "heat_rate": 0}, train, val)
    assert wrong["metrics"]["val"]["nmse"] > q.REFERENCE_NMSE_LIMIT
    assert PublicSplit.model_validate(frozen["training"]).name == "train"


def test_missing_reference_witness_does_not_launch_fit(audited, tmp_path, monkeypatch):
    release.publish_audited_release(*audited)
    root = tmp_path / "diagnostic"
    q.prepare(root, audited[-1], audited[1])
    monkeypatch.setattr(q, "replay", lambda *a: {"complete": False, "metrics": {}})
    monkeypatch.setattr(
        public, "execute_fit", lambda *a: pytest.fail("fit despite failed reference")
    )
    assert q.run_task(root, 0)["status"] == "reference_replay_failed"


@pytest.mark.parametrize("field", ["_runtime", "_source_identity"])
def test_execution_rejects_environment_drift(audited, tmp_path, monkeypatch, field):
    release.publish_audited_release(*audited)
    root = tmp_path / "diagnostic"
    q.prepare(root, audited[-1], audited[1])
    monkeypatch.setattr(public, field, lambda: "changed")
    with pytest.raises(ValueError, match="runtime or source changed"):
        q.run_task(root, 0)
    assert not (root / "cstr_easy_reference_near/fit/started.json").exists()
    assert q.report(root)["status"] == "incomplete"


def test_starts_are_declared_and_reference_assistance_is_explicit():
    specification = public._read(Path("data_raw") / q.SPEC_PATH)
    request, truth = q.cstr_request("hard", "reference_near", specification)
    assert request.source.stage == "synthetic_control"
    assert request.parameter_guesses["activation"] == 0.8 * truth["activation"]
    model, _, _ = public._lower(request)
    assert set(truth) == set(model.parameter_names)
    with pytest.raises(ValueError):
        q.cstr_request("hard", "best_validation", specification)
