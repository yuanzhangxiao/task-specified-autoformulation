"""Complete pairing, historical identity, and public-only import boundaries."""

import json
from copy import deepcopy

import pytest

from autoformalism.fitting.initialization_qualification import fixture
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research import construction_paired_inputs as paired
from autoformalism.staged_topology import content_hash
from scripts import construction_feedback_revision as revision


def synthetic_inputs():
    """Small public trajectory-root parent/child pair for campaign smoke tests."""
    example = fixture("trajectory_root", 0)
    problem = deepcopy(example["problem"])
    # The child is a separate saved proposal with the same scientific equations;
    # numerical tests may change equations without depending on an oracle truth.
    records = {
        "parent-00": {
            "role": "parent",
            "parent_id": "synthetic-parent",
            "parent_task_id": "parent-00",
            "case_id": "synthetic-parent",
            "problem": problem,
            "source_packet_sha256": "1" * 64,
        },
        "child-00": {
            "role": "child",
            "parent_id": "synthetic-parent",
            "parent_task_id": "parent-00",
            "case_id": "synthetic-revision",
            "problem": deepcopy(problem),
            "source_packet_sha256": "2" * 64,
        },
    }
    prefix = {"duration": 1, "future_inputs": []}
    data = {
        "protocol": paired.PROTOCOL,
        "records": records,
        "training": example["training"],
        "validation": example["validation"],
        "prefix_policy": prefix,
        "episodes": [
            {
                "case_id": "synthetic-revision",
                "parent_id": "synthetic-parent",
                "parent_task_id": "parent-00",
                "revision_seed": 0,
                "status": "revised",
                "completion_sha256": "3" * 64,
                "result_sha256": "4" * 64,
                "fit_task_id": "child-00",
            }
        ],
        "source": {
            "original_archive_sha256": "1" * 64,
            "original_plan_sha256": "2" * 64,
            "original_runtime_source_sha256": "3" * 64,
            "benchmark_id": "synthetic-public",
            "public_files": {},
            "forcing_authorization": "synthetic forcing",
            "permitted_future_inputs": [],
            "permitted_future_auxiliaries": [],
            "revision_plan_sha256": "4" * 64,
            "revision_inputs_sha256": "5" * 64,
            "revision_runtime_source_sha256": "6" * 64,
            "revision_entrypoints": {},
            "consistency_evidence_sha256": "7" * 64,
            "revision_members": {},
            "verifier_source_sha256": "8" * 64,
        },
        "test_data_opened": False,
    }
    return paired.validate_inputs(data, "synthetic-smoke")


def test_synthetic_boundary_retains_scope_and_same_data():
    data = synthetic_inputs()
    canonical = paired.validate_inputs(data, "synthetic-smoke")
    assert canonical == data
    assert (
        canonical["records"]["child-00"]["problem"]["graph"]
        == (canonical["records"]["parent-00"]["problem"]["graph"])
    )
    with pytest.raises(ValueError, match="8 parents"):
        paired.validate_inputs(data)


@pytest.mark.parametrize("role", ["inputs", "auxiliaries"])
def test_declared_but_absent_future_forcing_cannot_be_sealed(role):
    data = synthetic_inputs()
    data["prefix_policy"][f"future_{role}"] = ["unavailable"]
    data["source"][f"permitted_future_{role}"] = ["unavailable"]
    with pytest.raises(ValueError):
        paired.validate_inputs(data, "synthetic-smoke")


@pytest.mark.parametrize(
    "mutation",
    [
        "test",
        "extra",
        "overlap",
        "parent",
        "alias",
        "rejection",
        "lost-child",
        "duplicate-episode",
        "forcing",
        "suffix",
        "target",
        "scope",
        "case",
    ],
)
def test_boundary_rejects_unbound_or_leaking_inputs(mutation):
    data = synthetic_inputs()
    if mutation == "test":
        data["test_data_opened"] = True
    elif mutation == "extra":
        data["source"]["test_metrics"] = {"nmse": 0}
    elif mutation == "overlap":
        data["validation"]["rows"][0]["trajectory_id"] = data["training"]["rows"][0][
            "trajectory_id"
        ]
    elif mutation == "parent":
        data["records"]["child-00"]["parent_task_id"] = "missing"
    elif mutation == "alias":
        data["episodes"][0]["status"] = "retained"
    elif mutation == "rejection":
        data["episodes"][0]["status"] = "rejected"
    elif mutation == "lost-child":
        data["episodes"] = []
    elif mutation == "duplicate-episode":
        data["episodes"].append(deepcopy(data["episodes"][0]))
    elif mutation == "forcing":
        data["source"]["permitted_future_inputs"] = ["unlisted"]
    elif mutation == "suffix":
        data["prefix_policy"]["duration"] = 10000
    elif mutation == "target":
        data["training"]["rows"][0]["targets"]["unlisted"] = data["training"]["rows"][
            0
        ]["targets"]["y"]
    elif mutation == "scope":
        child = data["records"]["child-00"]["problem"]["graph"]
        next(iter(child["rules"].values()))["initial"]["scope"] = "shared"
    else:
        data["records"]["child-00"]["case_id"] = "synthetic-parent"
    with pytest.raises(ValueError):
        paired.validate_inputs(data, "synthetic-smoke")


@pytest.fixture
def historical(tmp_path, monkeypatch):
    """Sealed 24-episode import fixture; primitive replay has separate real tests."""
    data = synthetic_inputs()
    training, validation = deepcopy(data["training"]), deepcopy(data["validation"])
    training["rows"] = [
        {**deepcopy(training["rows"][0]), "trajectory_id": f"train-{i}"}
        for i in range(16)
    ]
    validation["rows"] = [
        {**deepcopy(validation["rows"][0]), "trajectory_id": f"val-{i}"}
        for i in range(4)
    ]
    for split in (training, validation):
        for row in split["rows"]:
            row["time"] = [10 * t for t in row["time"]]
    original = {
        "records": {
            f"model-{i}": {
                "problem": deepcopy(data["records"]["parent-00"]["problem"]),
                "source_packet_sha256": content_hash(i),
            }
            for i in range(8)
        },
        "training": training,
        "validation": validation,
        "source": {
            "archive_sha256": "1" * 64,
            "plan_sha256": "2" * 64,
            "historical_runtime_sha256": "3" * 64,
            "benchmark_id": "public-toy",
            "public_files": dict.fromkeys(
                ("specification.json", "proposer_prompt.txt", "train.json", "val.json"),
                "4" * 64,
            ),
            "forcing_authorization": paired.original._SUPPLIED_HORIZON,
            "permitted_future_inputs": [],
            "permitted_future_auxiliaries": [],
        },
    }
    projection = {
        "parents": deepcopy(original["records"]),
        "training": training,
        "fitting_prefix_policy": {"duration": 8, "future_inputs": []},
    }
    monkeypatch.setattr(revision, "_validate_inputs", lambda value: None)
    monkeypatch.setattr(paired.original, "load_inputs", lambda *a: deepcopy(original))
    monkeypatch.setattr(paired.consistency, "build", lambda *a: {"evidence": "saved"})
    root = tmp_path / "revisions"
    sealed_write(root / "inputs.json", projection)
    sidecar = sealed_write(root / "consistency_evidence.json", {"evidence": "saved"})
    cases = []
    for name, parent in sorted(projection["parents"].items()):
        sealed_write(
            root / "controls" / f"{name}.json",
            {
                "parent_id": name,
                "problem": parent["problem"],
                "problem_sha256": content_hash(parent["problem"]),
                "control_unchanged": True,
                "fitting_launched": False,
            },
        )
        for seed in revision.SEEDS:
            identity = {
                "parent_id": name,
                "seed": seed,
                "input_sha256": content_hash(parent),
            }
            cases.append(
                {
                    **identity,
                    "case_id": f"{name}__revision_seed{seed}",
                    "namespace": f"{revision.V2_PROTOCOL}:{content_hash(identity)}",
                }
            )
    plan = {
        "protocol": revision.V2_PROTOCOL,
        "cases": cases,
        "settings": revision.SETTINGS.model_dump(mode="json"),
        "input_bundle_sha256": content_hash(projection),
        "runtime_source_sha256": "a" * 64,
        "entrypoints": revision._entrypoints(),
        "selection": "all eight parents, three seeds each; unchanged controls exported",
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "model_promotion": False,
        "policies": revision.V2_POLICIES,
        "consistency_evidence_sha256": sidecar["artifact_sha256"],
    }
    sealed_write(root / "plan.json", plan)
    for index, entry in enumerate(cases):
        folder = root / "results" / entry["case_id"]
        status = "revised" if index < 16 else "retained" if index == 16 else "rejected"
        problem = projection["parents"][entry["parent_id"]]["problem"]
        sealed_write(folder / "completion.json", {"status": status})
        (folder / "result.json").write_text(
            json.dumps(
                {"status": status, "problem": problem if status != "rejected" else None}
            )
        )
        if status != "rejected":
            sealed_write(folder / "fit_handoff.json", {"prefix_fit_problem": problem})
    calls = []

    def replay(root, entry, plan):
        calls.append(entry["case_id"])
        return paired._Reader(root).json(
            f"results/{entry['case_id']}/completion.json", "artifact_sha256"
        )

    monkeypatch.setattr(revision, "_completion", replay)
    return root, original, calls


def test_import_preserves_old_hash_and_complete_population(historical, tmp_path):
    root, _, calls = historical
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.json")}
    # Deliberately unreadable evaluator-only material is never selected.
    (root / "test.json").write_text("NOT PUBLIC JSON")
    result = paired.load_inputs(root, tmp_path / "source.tar.gz", tmp_path / "public")
    assert len(calls) == len(result["episodes"]) == len(result["records"]) == 24
    assert result["source"]["revision_runtime_source_sha256"] == "a" * 64
    assert result["source"]["verifier_source_sha256"] != "a" * 64
    assert len(result["training"]["rows"]) == 16
    assert len(result["validation"]["rows"]) == 4
    retained = next(e for e in result["episodes"] if e["status"] == "retained")
    assert retained["fit_task_id"] == retained["parent_task_id"]
    assert "test.json" not in result["source"]["revision_members"]
    assert all((root / name).read_bytes() == blob for name, blob in before.items())


@pytest.mark.parametrize(
    "mutation", ["parent", "training", "sidecar", "control", "plan"]
)
def test_import_rejects_changed_source_before_numerics(historical, tmp_path, mutation):
    root, original, _ = historical
    if mutation == "parent":
        original["records"]["model-0"]["source_packet_sha256"] = "e" * 64
    elif mutation == "training":
        original["training"]["rows"][0]["targets"]["y"][0] += 1
    else:
        name = {
            "sidecar": "consistency_evidence.json",
            "control": "controls/model-0.json",
            "plan": "plan.json",
        }[mutation]
        path = root / name
        value = json.loads(path.read_text())
        del value["artifact_sha256"]
        if mutation == "sidecar":
            value["evidence"] = "changed"
        elif mutation == "control":
            value["control_unchanged"] = False
        else:
            value["settings"]["max_output_tokens"] += 1
        path.unlink()
        sealed_write(path, value)
    with pytest.raises(ValueError):
        paired.load_inputs(root, tmp_path / "source", tmp_path / "public")


def test_import_refuses_symlink_before_replay(historical, tmp_path):
    root, _, _ = historical
    target = next((root / "results").glob("*/result.json"))
    copy = tmp_path / "outside.json"
    copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(copy)
    with pytest.raises(ValueError, match="symlink"):
        paired.load_inputs(root, tmp_path / "source", tmp_path / "public")
