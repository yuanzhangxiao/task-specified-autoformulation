"""New preparation contracts, public information and release readiness."""

import pytest

from autoformalism.benchmarks.phase_b_generation import phase_b_protocols
from autoformalism.benchmarks.phase_c_protocols import phase_c_protocols


def test_alien_reset_preserves_history_and_varies_only_observed_output():
    before = phase_b_protocols("alien_device", input_contract="continuous-rates-1")
    snapshot = [p.model_dump(mode="json") for p in before]
    new = phase_c_protocols("alien_device")
    for old, current in zip(before, new, strict=True):
        assert old.input_contract == current.input_contract == "continuous-rates-1"
        assert {k: v for k, v in old.specification.items() if k != "initial_shift"} == {
            k: v for k, v in current.specification.items() if k != "initial_shift"
        }
        shift = current.specification.get("initial_shift", [0] * 6)
        assert shift[:-1] == [0] * 5
    assert new[14].specification["initial_shift"][-1] == 0.5
    assert new[15].specification["initial_shift"][-1] == -0.5
    assert snapshot == [
        p.model_dump(mode="json")
        for p in phase_b_protocols("alien_device", input_contract="continuous-rates-1")
    ]
    assert before[14].specification["initial_shift"][0] == 0.5


def test_cstr_hidden_preparation_shared_in_every_split():
    rows = phase_c_protocols("cstr")
    assert len(rows) == 26
    for row in rows:
        shift = row.specification.get("initial_shift", [0, 0, 0])
        assert shift[0] == shift[2] == 0
    assert rows[15].specification["initial_shift"][1] == 5
    assert rows[-2].specification["initial_shift"][1] == 8
    assert rows[-1].specification["initial_shift"][1] == -6


@pytest.mark.parametrize("task", ["T1", "T2"])
def test_dalla_keeps_existing_continuous_inputs_and_preparation(task):
    before = phase_b_protocols(
        "dalla_man", task=task, input_contract="continuous-rates-1"
    )
    assert phase_c_protocols("dalla_man", task=task) == before
    assert sum(p.split == "train" for p in before) == 16
    assert sum(p.split == "validation" for p in before) == 4
    assert any(len(p.specification.get("meals", [])) > 1 for p in before)
    assert all(p.input_names[0] == "meal_rate_g_per_min" for p in before)


@pytest.mark.parametrize("task", [None, "T3", "T4"])
def test_dalla_requires_an_explicit_in_scope_task(task):
    with pytest.raises(ValueError, match="only explicit Dalla T1/T2"):
        phase_c_protocols("dalla_man", task=task)


def test_public_information_audit_detects_hidden_preparation_not_ids():
    from autoformalism.benchmarks.phase_c_release import public_information_audit
    from autoformalism.schemas.public_fitting import PublicSplit

    def split(second_initial=0, second_input=0, second_future=2):
        return PublicSplit.model_validate(
            {
                "name": "train",
                "fingerprint": "test",
                "rows": [
                    {
                        "trajectory_id": "a",
                        "time": [0, 1],
                        "targets": {"y": [0, 1]},
                        "external_inputs": {"u": [0, 0]},
                    },
                    {
                        "trajectory_id": "different-id",
                        "time": [0, 1],
                        "targets": {"y": [second_initial, second_future]},
                        "external_inputs": {"u": [0, second_input]},
                    },
                ],
            }
        )

    assert not public_information_audit(split())["passed"]
    assert public_information_audit(split(second_initial=0.5))["passed"]
    assert public_information_audit(split(second_input=1))["passed"]
    assert public_information_audit(split(second_future=1))["passed"]


@pytest.fixture(params=["alien_device", "cstr"])
def small_release(tmp_path, monkeypatch, request):
    from pathlib import Path

    from autoformalism.benchmarks import phase_c_release as release

    family = request.param
    data = Path("data_raw")
    if not (data / release.SPEC_PATHS[family]).exists():
        pytest.skip("private evaluator specification not installed")
    base = phase_c_protocols(family)
    selected = [base[0], base[15], base[16]]
    short = tuple(
        p.model_copy(
            update={
                "duration": 0.5,
                "dt": 0.1,
                "input_dt": 0.1,
                "specification": {**p.specification, "start": 0.1, "end": 0.3}
                if p.specification["kind"] in {"step", "pulse"}
                else p.specification,
            }
        )
        for p in selected
    )
    monkeypatch.setattr(release, "phase_c_protocols", lambda _: short)
    root = tmp_path / "release"
    report = release.build(root, data, (family,))
    return root, data, family, report


def test_public_release_known_equations_and_immutable_resume(
    small_release, monkeypatch
):
    from autoformalism.benchmarks import phase_c_release as release
    from autoformalism.benchmarks.audited_release import read_seal

    root, data, family, report = small_release
    assert report["ready_cells"] == report["development_cells"] == 4
    assert not report["whole_phase_c_roster_ready"]
    assert not report["test_generated"] and not report["parameter_fitting_performed"]
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        release, "audit_one", lambda *a: pytest.fail("regenerated completed data")
    )
    assert release.build(root, data, (family,)) == report
    assert release.verify(root) == report
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    for directory in (root / "public").iterdir():
        specification = read_seal(directory / "specification.json")
        assert "preparation_contract" in specification
        assert "truth" not in specification and "private_source" not in str(
            specification
        )
        assert {p.name for p in directory.iterdir()} == {
            "train.json",
            "val.json",
            "specification.json",
            "proposer_prompt.txt",
        }
    assert len({r["numeric_identity"] for r in report["cells"][:2]}) == 1
    assert len({r["numeric_identity"] for r in report["cells"][2:]}) == 1


@pytest.mark.parametrize("kind", ["public", "private"])
def test_release_tampering_detected(small_release, kind):
    from autoformalism.benchmarks import phase_c_release as release

    root, _, _, _ = small_release
    path = next(
        (root / ("public" if kind == "public" else "diagnostic")).rglob("*.json")
    )
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="release file changed"):
        release.verify(root)


def test_source_drift_cannot_resume(small_release, monkeypatch):
    from autoformalism.benchmarks import phase_c_release as release

    root, data, family, _ = small_release
    monkeypatch.setattr(release.public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        release.build(root, data, (family,))


def test_partial_resume_verifies_reference_arrays(small_release):
    from autoformalism.benchmarks import phase_c_release as release

    root, data, family, report = small_release
    # An interruption after a family checkpoint must not bless a changed origin.
    (root / "summary.json").unlink()
    assert release.build(root, data, (family,)) == report
    (root / "summary.json").unlink()
    array = next((root / "diagnostic" / "references").rglob("*.npz"))
    array.write_bytes(array.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="release file changed"):
        release.build(root, data, (family,))


def test_failed_gate_cannot_mark_release_ready(tmp_path, monkeypatch):
    from pathlib import Path

    from autoformalism.benchmarks import phase_c_release as release

    monkeypatch.setattr(
        release,
        "_detention_family",
        lambda _: [
            {"cell": "failed_control", "ready_for_development": False, "files": {}}
        ],
    )
    report = release.build(tmp_path / "failed", Path("data_raw"), ("detention",))
    assert report["status"] == "complete"
    assert report["development_cells"] == 1 and report["ready_cells"] == 0
    assert not report["ready_for_development"]


def test_public_data_identifiers_cannot_encode_private_preparation():
    from pathlib import Path

    from autoformalism.benchmarks.phase_b_generation import simulate_phase_b
    from autoformalism.benchmarks.phase_b_public import phase_b_public_spec
    from autoformalism.benchmarks.phase_c_release import (
        SPEC_PATHS,
        project_split,
        public_information_audit,
    )

    if not (Path("data_raw") / SPEC_PATHS["alien_device"]).exists():
        pytest.skip("private evaluator specification not installed")
    spec = phase_b_public_spec(
        "alien_device", "hard", "functional", input_contract="continuous-rates-1"
    )
    historical = phase_b_protocols("alien_device", input_contract="continuous-rates-1")
    new = phase_c_protocols("alien_device")

    def projection(protocols):
        return project_split(
            spec,
            tuple(
                simulate_phase_b(
                    protocols[i].model_copy(
                        update={"duration": 2, "dt": 0.1, "input_dt": 0.1}
                    )
                )
                for i in (0, 14, 15)
            ),
            "train",
        )

    assert not public_information_audit(projection(historical))["passed"]
    assert public_information_audit(projection(new))["passed"]


def test_public_loader_never_reads_evaluator_files(small_release, monkeypatch):
    from pathlib import Path

    from autoformalism.benchmarks.phase_c_release import load_public_cell

    root, _, _, report = small_release
    directory = root / "public" / report["cells"][0]["cell"]
    original = Path.open

    def public_only(self, *args, **kwargs):
        assert self.is_relative_to(directory), f"evaluator path opened: {self}"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", public_only)
    spec, training, validation = load_public_cell(directory)
    assert training.name == "train" and validation.name == "val"
    assert not spec["test_released"]


def test_basin_is_included_with_public_initials_and_noise_controls(tmp_path):
    from pathlib import Path

    from autoformalism.benchmarks.phase_c_release import build, load_public_cell, verify

    root = tmp_path / "basin"
    report = build(root, Path("data_raw"), ("detention",))
    assert report["ready_cells"] == 4
    assert report == verify(root)
    for row in report["cells"]:
        spec, train, val = load_public_cell(root / "public" / row["cell"])
        assert (len(train.rows), len(val.rows)) == (6, 3)
        assert "initial_up" in spec["fixed_covariates"]
        assert "initial_up" in train.rows[0].fixed_covariates
        assert spec["negative_control"] == ("independent" in row["cell"])
        assert spec["noise_sd_fraction"] in (0, 0.01)


@pytest.fixture
def small_dalla_release(tmp_path, monkeypatch):
    from pathlib import Path

    from autoformalism.benchmarks import phase_c_release as release

    def small_protocols(family, *, task):
        base = phase_c_protocols(family, task=task)
        # Retain a multiple-meal schedule and the validation initial shift.
        # The reserved test protocol has out-of-horizon events after shortening;
        # accidentally generating it must fail rather than go unnoticed.
        short = [base[0], base[9], base[19], base[-1]]
        short[1] = short[1].model_copy(
            update={"specification": {"meals": [[0, 30], [10, 30]]}}
        )
        return tuple(p.model_copy(update={"duration": 20}) for p in short)

    monkeypatch.setattr(release, "phase_c_protocols", small_protocols)
    monkeypatch.setattr(
        release,
        "reference_request",
        lambda *a: pytest.fail("Dalla exact private transcription is not a gate"),
    )
    monkeypatch.setattr(
        release,
        "replay",
        lambda *a: pytest.fail("no exact Dalla replay claimed"),
    )
    root = tmp_path / "dalla"
    report = release.build(root, Path("data_raw"), ("dalla_man",))
    return root, report


def test_dalla_scope_public_projection_and_immutable_resume(small_dalla_release):
    from pathlib import Path

    from autoformalism.benchmarks import phase_c_release as release
    from autoformalism.benchmarks.audited_release import read_seal
    from autoformalism.benchmarks.phase_b_public import (
        phase_b_public_spec,
        render_phase_b_prompts,
    )

    root, report = small_dalla_release
    assert report["development_cells"] == report["ready_cells"] == 16
    assert report["ready_for_development"]
    assert not report["whole_phase_c_roster_ready"]
    assert not report["test_generated"]
    assert not report["parameter_fitting_performed"] and report["llm_calls"] == 0
    assert not report["dalla_model_class_scope"][
        "exact_reference_transcription_required"
    ]
    names = []
    for row in report["cells"]:
        directory = root / "public" / row["cell"]
        spec, train, val = release.load_public_cell(directory)
        names.append(row["cell"])
        task = "T1" if "_t1_" in row["cell"] else "T2"
        tier = "easy" if "_easy_" in row["cell"] else "hard"
        variant = "obfuscated" if "_obfuscated_" in row["cell"] else "named"
        dynamics = "perturbed" if "_perturbed_" in row["cell"] else "canonical"
        original = phase_b_public_spec(
            "dalla_man",
            tier,
            variant,
            task=task,
            dynamics=dynamics,
            input_contract="continuous-rates-1",
        )
        assert spec["public_prompt"] == render_phase_b_prompts(original)[0]
        assert len(train.rows) == 2 and len(val.rows) == 1
        for forbidden in ("Qsto", "meal_reference", "D_ref", "private_initial"):
            assert forbidden not in str(spec)
        assert len(spec["targets"]) == (
            1 if task == "T1" else 3 if tier == "easy" else 2
        )
        audit = read_seal(root / "diagnostic" / row["cell"] / "audit.json")
        assert audit["passed"] and audit["initially_empty_gut_verified"]
        assert audit["reference_numerics_passed"]
        assert not audit["public_interface_replay"]["performed"]
        assert not audit["shared_hidden_preparation_asserted"]
        assert "Il" in audit["private_initial_coordinate_ranges"]
    assert len(set(names)) == 16
    for index in range(0, 16, 2):
        assert (
            report["cells"][index]["numeric_identity"]
            == report["cells"][index + 1]["numeric_identity"]
        )
    references = list((root / "diagnostic" / "references").rglob("*.npz"))
    assert len(references) == 12 and all("test_" not in p.name for p in references)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert release.build(root, Path("data_raw"), ("dalla_man",)) == report
    assert release.verify(root) == report
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (root / "summary.json").unlink()
    assert release.build(root, Path("data_raw"), ("dalla_man",)) == report
    (root / "summary.json").unlink()
    references[0].write_bytes(references[0].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="release file changed"):
        release.build(root, Path("data_raw"), ("dalla_man",))


@pytest.mark.parametrize("defect", [None, "missing", "duplicate", "failed"])
def test_whole_roster_readiness_requires_every_cell(tmp_path, monkeypatch, defect):
    from pathlib import Path

    from autoformalism.benchmarks import phase_c_release as release

    def rows(family):
        result = [
            {"cell": f"{family}_{i}", "ready_for_development": True, "files": {}}
            for i in range(release.CELL_COUNTS[family])
        ]
        if family == "dalla_man":
            if defect == "missing":
                result.pop()
            elif defect == "duplicate":
                result[-1] = result[0]
            elif defect == "failed":
                result[-1]["ready_for_development"] = False
        return result

    monkeypatch.setattr(release, "_detention_family", lambda _: rows("detention"))
    monkeypatch.setattr(release, "_phase_b_family", lambda _, f, *a: rows(f))
    monkeypatch.setattr(release, "_dalla_family", lambda *a: rows("dalla_man"))
    report = release.build(tmp_path / "roster", Path("data_raw"))
    assert report["expected_cells"] == 28
    assert report["whole_phase_c_roster_ready"] is (defect is None)
    assert report["ready_for_development"] is (defect is None)


def test_legacy_phase_c_public_cells_remain_loadable(small_release):
    from autoformalism.benchmarks import phase_c_release as release
    from autoformalism.benchmarks.audited_release import read_seal, seal

    root, _, _, report = small_release
    directory = root / "public" / report["cells"][0]["cell"]
    path = directory / "specification.json"
    old = read_seal(path)
    old["protocol"] = "phase-c-development-1"
    path.unlink()
    seal(path, old)
    assert release.load_public_cell(directory)[0]["protocol"] == "phase-c-development-1"


def test_dalla_numerical_failure_still_blocks_release(tmp_path, monkeypatch):
    from pathlib import Path

    from autoformalism.benchmarks import phase_c_release as release

    monkeypatch.setattr(release, "audit_one", lambda *a: ({"passed": False}, None))
    root = tmp_path / "failed-numerics"
    with pytest.raises(ValueError, match="Dalla reference numerical audit failed"):
        release.build(root, Path("data_raw"), ("dalla_man",))
    assert not (root / "summary.json").exists()
    assert not (root / "diagnostic/dalla_man_complete.json").exists()
