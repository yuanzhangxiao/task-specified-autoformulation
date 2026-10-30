"""Byte authentication, full-roster preservation and public-only fitting inputs."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from autoformalism.research import construction_portfolio_intake as intake
from autoformalism.schemas.prefix_initialization import PrefixFitProblem


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text())


def _problem(index: int) -> dict:
    """A small public model includes ongoing auxiliary forcing and a free root."""
    return PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": f"model_{index}",
                "parent_candidate_id": None,
                "states": [
                    {"name": "x", "kind": "observed"},
                    {"name": "z", "kind": "latent"},
                ],
                "state_equations": [
                    {"state": "x", "rhs": "-k*x+z+Uii"},
                    {"state": "z", "rhs": "-z+u"},
                ],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "parameters": [
                    {
                        "name": "k",
                        "scope": "global",
                        "role": "rate",
                        "bounds": {"lower": 0.1, "upper": 2},
                    }
                ],
                "initial_conditions": [
                    {"state": state, "scope": "global", "fixed_value": 0}
                    for state in ("x", "z")
                ],
            },
            "context": {
                "targets": ["y"],
                "auxiliaries": ["Uii"],
                "external_inputs": ["u"],
            },
            "parameter_guesses": {"k": 0.5},
            "graph": {
                "rules": {
                    "z": {
                        "initial": {
                            "mode": "value",
                            "scope": "trajectory",
                            "guess": 0.3,
                        }
                    }
                }
            },
        }
    ).model_dump(mode="json")


def make_delivery(root: Path, count: int = 2) -> Path:
    """Create a complete synthetic delivery, independent of research artifacts."""
    root.mkdir(parents=True, exist_ok=True)
    prefix = {"duration": 1.0, "future_inputs": ["u"], "future_auxiliaries": ["Uii"]}
    training = {
        "name": "train",
        "fingerprint": "synthetic-only",
        "rows": [
            {
                "trajectory_id": f"train_{i}",
                "time": [0.0, 1.0, 2.0],
                "targets": {"y": [1.0 + i, 1.1 + i, 1.2 + i]},
                "auxiliaries": {"Uii": [1.0, 2.0, 3.0]},
                "external_inputs": {"u": [0.0, 1.0, 0.0]},
                "fixed_covariates": {},
            }
            for i in range(2)
        ],
    }
    _write(root / "evidence/training.json", training)
    _write(
        root / "evidence/public-context.json",
        {
            "training_sha256": intake.content_sha256(training),
            "fixed_fitting_prefix_policy": prefix,
        },
    )
    _write(root / "evidence/compatibility.json", {"source": "synthetic-public"})
    (root / "SOURCE_COMMIT").write_text("a" * 40 + "\n")
    roster = []
    for index in range(count):
        name = f"candidate-{index:02d}"
        problem = _problem(index)
        family = intake.content_sha256({"family": index})
        record = f"source-{index}"
        packet = {
            "protocol": "scaled-readout-fitting-intake-1",
            "candidate_id": name,
            "family": family,
            "problem": problem,
            "audit": {},
            "records": [
                {
                    "record": record,
                    "current_status": "eligible",
                    "family": family,
                    "source_result": "/must/not/follow/embedded/source/path.json",
                }
            ],
            "static_interface": {"status": "static_interface_passed"},
            "fixed_fitting_prefix_policy": prefix,
            "training": "../evidence/training.json",
            "compatibility_evidence": "../evidence/compatibility.json",
            "public_context": "../evidence/public-context.json",
            "fit_started": False,
            "scientific_validity": "not_established",
            "test_data_opened": False,
        }
        path = root / "models" / f"{name}.json"
        _write(path, packet)
        roster.append(
            {
                "candidate_id": name,
                "family": family,
                "path": f"models/{name}.json",
                "sha256": _digest(path),
                "problem_sha256": intake.content_hash(problem),
                "records": [record],
                "static_interface": "passed",
                "numerical_qualification": "pending",
            }
        )
    _write(
        root / "replay-summary.json",
        {
            "intake": roster,
            "unique_models": count,
            "static_interface_passed": count,
            "test_data_opened": False,
        },
    )
    _write(
        root / "manifest.json",
        {
            "protocol": "scaled-readout-orion-delivery-1",
            "source_commit": "a" * 40,
            "unique_models": count,
            "static_interface_passed": count,
            "numerical_qualification": "pending",
            "optimizer_calls": 0,
            "llm_calls": 0,
            "solver_rollouts": 0,
            "test_data_opened": False,
            "files": {
                str(path.relative_to(root)): _digest(path)
                for path in sorted(root.rglob("*"))
                if path.is_file()
            },
        },
    )
    return root


def resign(root: Path) -> None:
    """Re-sign bytes so semantic adversarial tests reach boundary validation."""
    manifest = _read(root / "manifest.json")
    manifest["files"] = {name: _digest(root / name) for name in manifest["files"]}
    _write(root / "manifest.json", manifest)


def change_packet(root: Path, name: str, mutate) -> None:
    path = root / "models" / f"{name}.json"
    packet = _read(path)
    mutate(packet)
    _write(path, packet)
    summary = _read(root / "replay-summary.json")
    row = next(row for row in summary["intake"] if row["candidate_id"] == name)
    row["sha256"] = _digest(path)
    _write(root / "replay-summary.json", summary)
    resign(root)


def test_delivery_preserves_every_model_domain_graph_and_auxiliary(tmp_path):
    source = make_delivery(tmp_path / "source")
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    data = intake.load_delivery(source, expected_count=2)
    assert data["protocol"] == intake.PROTOCOL
    assert len(data["models"]) == 2
    assert "validation" not in data and not data["validation_exported"]
    assert not data["test_data_opened"]
    assert data["prefix"]["future_auxiliaries"] == ["Uii"]
    assert data["training"]["rows"][0]["auxiliaries"]["Uii"] == [1, 2, 3]
    for index, model in enumerate(data["models"]):
        assert model["problem"] == _problem(index)
        assert model["source_problem_sha256"] != model["fitting_problem_sha256"]
        assert model["records"] == [f"source-{index}"]
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert intake.load_delivery(source, expected_count=2) == data
    intake.validate_inputs(data, expected_count=2)


def test_real_roster_is_pinned_and_cannot_be_replaced_by_small_fixture(tmp_path):
    source = make_delivery(tmp_path / "source")
    with pytest.raises(ValueError, match="reviewed 23-model"):
        intake.load_delivery(source)
    with pytest.raises(ValueError, match="complete expected roster"):
        intake.load_delivery(source, expected_count=1)


@pytest.mark.parametrize("name", ["evidence/training.json", "models/candidate-00.json"])
def test_modified_bytes_fail_before_import(name, tmp_path):
    source = make_delivery(tmp_path / "source")
    with (source / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="digest differs"):
        intake.load_delivery(source, expected_count=2)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.json",
        "/tmp/outside.json",
        "evidence/test.json",
        "models/../outside.json",
    ],
)
def test_manifest_paths_cannot_escape_or_add_test_data(name, tmp_path):
    source = make_delivery(tmp_path / "source")
    manifest = _read(source / "manifest.json")
    manifest["files"][name] = "a" * 64
    _write(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="unsafe delivery manifest"):
        intake.load_delivery(source, expected_count=2)


def test_symlinked_file_and_parent_are_rejected(tmp_path):
    source = make_delivery(tmp_path / "source")
    packet = source / "models/candidate-00.json"
    outside = tmp_path / "outside.json"
    packet.rename(outside)
    packet.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        intake.load_delivery(source, expected_count=2)
    packet.unlink()
    outside.rename(packet)
    model_dir = source / "models"
    model_dir.rename(tmp_path / "models")
    model_dir.symlink_to(tmp_path / "models", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        intake.load_delivery(source, expected_count=2)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate_id",
        "duplicate_family",
        "duplicate_problem",
        "wrong_source_hash",
    ],
)
def test_incomplete_or_mismatched_roster_fails(mutation, tmp_path):
    source = make_delivery(tmp_path / "source")
    summary = _read(source / "replay-summary.json")
    rows = summary["intake"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate_id":
        rows[1]["candidate_id"] = rows[0]["candidate_id"]
    elif mutation == "duplicate_family":
        rows[1]["family"] = rows[0]["family"]
    elif mutation == "duplicate_problem":
        rows[1]["problem_sha256"] = rows[0]["problem_sha256"]
    else:
        rows[0]["problem_sha256"] = "a" * 64
    _write(source / "replay-summary.json", summary)
    resign(source)
    with pytest.raises(ValueError):
        intake.load_delivery(source, expected_count=2)


def test_unlisted_model_cannot_be_silently_ignored(tmp_path):
    source = make_delivery(tmp_path / "source")
    _write(source / "models/candidate-99.json", {})
    with pytest.raises(ValueError, match="unlisted candidate"):
        intake.load_delivery(source, expected_count=2)


@pytest.mark.parametrize(
    "mutation",
    [
        "prefix",
        "source_path",
        "fit_started",
        "rejected",
        "hidden_payload",
        "bad_digest",
    ],
)
def test_packet_contract_is_not_inferred_from_filename(mutation, tmp_path):
    source = make_delivery(tmp_path / "source")

    def mutate(packet):
        if mutation == "prefix":
            packet["fixed_fitting_prefix_policy"]["duration"] = 0
        elif mutation == "source_path":
            packet["training"] = "/some/other/training.json"
        elif mutation == "fit_started":
            packet["fit_started"] = True
        elif mutation == "rejected":
            packet["records"][0]["current_status"] = "rejected"
        elif mutation == "hidden_payload":
            packet["problem"]["test_data"] = {"y": [10]}
        else:
            packet["family"] = "not-a-digest"

    change_packet(source, "candidate-00", mutate)
    with pytest.raises(ValueError):
        intake.load_delivery(source, expected_count=2)


@pytest.mark.parametrize(
    "mutation",
    ["val", "hidden_states", "targets", "suffix", "missing_auxiliary", "wrong_digest"],
)
def test_only_complete_training_arrays_can_enter_fitting(mutation, tmp_path):
    source = make_delivery(tmp_path / "source")
    path = source / "evidence/training.json"
    training = _read(path)
    if mutation == "val":
        training["name"] = "val"
    elif mutation == "hidden_states":
        training["rows"][0]["hidden_states"] = {"z": [1, 2, 3]}
    elif mutation == "targets":
        training["rows"][0]["targets"] = {"other": [1, 2, 3]}
    elif mutation == "suffix":
        training["rows"][0]["time"] = [0, 0.1, 0.2]
    elif mutation == "missing_auxiliary":
        training["rows"][0]["auxiliaries"] = {}
    else:
        training["fingerprint"] = "changed"
    _write(path, training)
    if mutation != "wrong_digest":
        context = _read(source / "evidence/public-context.json")
        context["training_sha256"] = intake.content_sha256(training)
        _write(source / "evidence/public-context.json", context)
    resign(source)
    with pytest.raises(ValueError):
        intake.load_delivery(source, expected_count=2)


def test_duplicate_json_keys_and_oversized_files_fail_closed(tmp_path, monkeypatch):
    source = make_delivery(tmp_path / "source")
    original = (source / "manifest.json").read_bytes()
    payload = original.decode().replace(
        '"unique_models": 2', '"unique_models": 1, "unique_models": 2'
    )
    (source / "manifest.json").write_text(payload)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        intake.load_delivery(source, expected_count=2)
    (source / "manifest.json").write_bytes(original)
    monkeypatch.setattr(intake, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="bounded regular file"):
        intake.load_delivery(source, expected_count=2)


@pytest.mark.parametrize(
    "mutation",
    ["extra_split", "model", "training", "roster", "duplicate_record", "test_flag"],
)
def test_sealed_validation_requires_no_source_and_rejects_changes(mutation, tmp_path):
    source = make_delivery(tmp_path / "source")
    data = deepcopy(intake.load_delivery(source, expected_count=2))
    if mutation == "extra_split":
        data["validation"] = data["training"]
    elif mutation == "model":
        data["models"][0]["problem"]["parameter_guesses"]["k"] = 0.7
    elif mutation == "training":
        data["training"]["rows"][0]["targets"]["y"][0] += 1
    elif mutation == "roster":
        data["models"].reverse()
    elif mutation == "duplicate_record":
        data["models"][1]["records"] = data["models"][0]["records"]
    else:
        data["test_data_opened"] = True
    with pytest.raises(ValueError):
        intake.validate_inputs(data, expected_count=2)
