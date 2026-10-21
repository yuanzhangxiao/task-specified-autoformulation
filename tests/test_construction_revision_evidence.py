"""Training feedback cannot import held-out fields or alter saved endpoints."""

import hashlib
import io
import json
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.research import construction_revision_evidence as evidence
from tests.test_construction_fit_qualification import eight_inputs
from tests.test_construction_verification_campaign import file_bytes, reseal


def _fake_fit(problem, training, numerical, execution, checkpoint):
    """Synthetic journal fixture with a retained endpoint; no optimizer calls."""
    oracle = evidence.original.kernel.TrainingOracle(problem, training, numerical)
    parameters = dict(oracle.start)
    value = {
        "fit_success": False,
        "parameters": None,
        "problem_sha256": evidence.public.content_sha256(problem),
        "training_sha256": evidence.public.content_sha256(training),
        "policy": numerical.model_dump(mode="json"),
        "training_scales": oracle.scales,
        "verification_error": "historical numerical budget exhausted",
        "search": {
            "best": {"parameters": parameters, "training_nmse": 10.0},
            "evaluations": [
                {
                    "parameters_sha256": evidence.public.content_sha256(parameters),
                    "training_nmse": 10.0,
                }
            ],
            "stop_reason": "budget_limited",
        },
        "verification_timing": {"timeout_reason": "numerical_active_wall_limit"},
    }
    checkpoint(value)
    return value


def _fake_verifier(counter):
    """Record known independent arithmetic predictions, with one unavailable case."""

    def verify(problem, training, parameters, numerical, execution, checkpoint):
        counter.append(1)
        success = len(counter) % 8 != 0
        channels = tuple(training.rows[0].targets)
        scales = {
            name: max(
                float(np.std(np.concatenate([r.targets[name] for r in training.rows]))),
                1e-12,
            )
            for name in channels
        }
        predictions, row_scores, total = {}, {}, {name: [] for name in channels}
        for index, row in enumerate(training.rows):
            error = np.arange(len(row.time)) / 1000 + index / 100
            predictions[row.trajectory_id] = {
                name: (np.asarray(row.targets[name]) + error).tolist()
                for name in channels
            }
            scores = {
                name: float(np.mean((error / scales[name]) ** 2)) for name in channels
            }
            row_scores[row.trajectory_id] = {
                "sample_count": len(row.time),
                "normalized_mse": float(np.mean(list(scores.values()))),
                "per_target_normalized_mse": scores,
            }
            for name in channels:
                total[name].extend((error / scales[name]) ** 2)
        scores = {name: float(np.mean(values)) for name, values in total.items()}
        value = {
            "protocol": "construction-saved-parameter-verification-1",
            "success": success,
            "fit_success": success,
            "parameters": parameters if success else None,
            "parameters_sha256": evidence.public.content_sha256(parameters),
            "problem_sha256": evidence.public.content_sha256(problem),
            "training_sha256": evidence.public.content_sha256(training),
            "policy": numerical.model_dump(mode="json"),
            "execution_policy": execution.model_dump(mode="json"),
            "selection_data": "train_only",
            "sensitivities": False,
            "no_search_performed": True,
            "optimizer_calls": 0,
            "live_llm_calls": 0,
            "test_data_opened": False,
            "private_data_opened": False,
            "verification_calls_completed": 2 if success else 0,
            "rollout_calls_completed": 2 * len(training.rows) if success else 1,
            "verification_calls_failed": 0 if success else 1,
            "rollout_calls_failed": 0 if success else 1,
            "normalized_mse": float(np.mean(list(scores.values())))
            if success
            else None,
            "per_target_normalized_mse": scores if success else {},
            "training_scales": scales,
            "predictions": predictions if success else {},
            "per_row_scores": row_scores if success else {},
            "solver_difference": 1e-10 if success else None,
            "timing": {
                "timeout_reason": None if success else "numerical_active_wall_limit"
            },
        }
        checkpoint(value)
        return value

    return verify


def _build(root: Path, *, poison=987654321.0):
    """Build sealed real campaign formats with inexpensive synthetic receipts."""
    source, supplemental = root / "source", root / "supplemental"
    root.mkdir(parents=True, exist_ok=True)
    prompt = root / "public_prompt.txt"
    prompt.write_text("Public synthetic mechanistic task.\n")
    archive = root / "construction.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("placeholder.txt")
        member.size = 4
        stream.addfile(member, io.BytesIO(b"test"))
    data = eight_inputs()
    for row in data["validation"]["rows"]:
        row["targets"]["y"] = [poison] * len(row["time"])
        row["trajectory_id"] = row["trajectory_id"] + "_FORBIDDEN_HELDOUT"
    data["source"].update(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        plan_sha256="a" * 64,
        benchmark_id="public-synthetic",
        public_files={
            "proposer_prompt.txt": hashlib.sha256(prompt.read_bytes()).hexdigest()
        },
        unprojected_metadata="FORBIDDEN_SOURCE_METADATA",
    )
    for record in data["records"].values():
        record["metadata"]["unprojected"] = "FORBIDDEN_RECORD_METADATA"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evidence.public, "_source_identity", lambda: "synthetic-source")
        patch.setattr(
            evidence.campaign, "_entrypoints", lambda: {"synthetic": "b" * 64}
        )
        patch.setattr(evidence.original.execution, "fit_training", _fake_fit)
        patch.setattr(
            evidence.campaign.numerical, "verify_saved_parameters", _fake_verifier([])
        )
        rules = evidence.original.Policy(
            prefix=evidence.original.PrefixPolicy(duration=1)
        )
        evidence.original.prepare(source, data, rules)
        for index in range(8):
            evidence.original.run_task(source, index)
        evidence.campaign.prepare(supplemental, source)
        for index in range(8):
            evidence.campaign.run_task(supplemental, index)
    return source, supplemental, prompt, archive


@pytest.fixture(scope="module")
def originals(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("revision-evidence"))


@pytest.fixture
def paths(tmp_path, originals):
    source, supplemental, prompt, archive = originals
    shutil.copytree(source, tmp_path / "source")
    shutil.copytree(supplemental, tmp_path / "supplemental")
    return tmp_path / "source", tmp_path / "supplemental", prompt, archive


def test_all_parents_are_projected_without_heldout_metadata_or_new_numerics(
    originals, monkeypatch
):
    source, supplemental, prompt, _ = originals
    before = (file_bytes(source), file_bytes(supplemental))

    def forbidden(*args, **kwargs):
        pytest.fail("new numerical work or provider call")

    for name in ("TrainingOracle", "symbolic_rollout", "fit_training", "infer_prefix"):
        monkeypatch.setattr(evidence.original.kernel, name, forbidden)
    monkeypatch.setattr(evidence.original.kernel.recovery_numerics, "search", forbidden)
    result = evidence.load_training_evidence(source, supplemental, public_prompt=prompt)
    assert len(result["parents"]) == 8
    assert list(result["parents"]) == [f"model_{index}" for index in range(8)]
    assert len(result["training"]["rows"]) == 16
    assert len(result["training_summary"]["initial_observations"]) == 16
    text = json.dumps(result, allow_nan=False)
    assert "FORBIDDEN" not in text and "987654321" not in text
    assert '"validation"' not in text and "historical_validation_score" not in text
    assert result["public_task"]["prompt_text"] == prompt.read_text()
    assert (
        result["optimizer_calls"]
        == result["numerical_rollouts"]
        == result["llm_calls"]
        == 0
    )
    assert not result["test_data_opened"]
    evidence.validate_training_projection(result)
    for name, parent in result["parents"].items():
        fit = read_seal(source / f"results/{name}/operations/training/result.json")[
            "value"
        ]
        assert parent["endpoint"]["parameters"] == fit["search"]["best"]["parameters"]
        assert parent["prompt_evidence"]["search"]["budget_exhausted"]
        assert not parent["prompt_evidence"]["search"]["best_attainable_fit_assessed"]
        assert parent["public_brief"] is None
        assert parent["scientific_aliases"] == {}
        assert parent["scientific_aliases_sha256"] == evidence.public.content_sha256({})
        assert "metadata" not in parent
    assert before == (file_bytes(source), file_bytes(supplemental))
    assert (
        evidence.load_training_evidence(source, supplemental, public_prompt=prompt)
        == result
    )


def test_witnesses_and_initials_match_exact_training_rows(originals):
    result = evidence.load_training_evidence(*originals[:2])
    parent = result["parents"]["model_0"]
    target = parent["prompt_evidence"]["per_target"]["y"]
    witnesses = target["witnesses"]
    assert len(witnesses) == 3
    assert [(w["trajectory_id"], w["sample_index"]) for w in witnesses] == [
        ("training_15", 40),
        ("training_15", 39),
        ("training_15", 38),
    ]
    rows = {row["trajectory_id"]: row for row in result["training"]["rows"]}
    for witness in witnesses:
        row = rows[witness["trajectory_id"]]
        index = witness["sample_index"]
        assert witness["observed"] == row["targets"]["y"][index]
        assert witness["time"] == row["time"][index]
        assert witness["signed_error"] == pytest.approx(0.15 + index / 1000)
        assert witness["predicted"] == pytest.approx(
            witness["observed"] + witness["signed_error"]
        )
    for boundary in result["training_summary"]["initial_observations"]:
        row = rows[boundary["trajectory_id"]]
        assert boundary["targets"] == {
            name: values[0] for name, values in row["targets"].items()
        }
    failed = result["parents"]["model_7"]
    assert failed["prompt_evidence"]["status"] == "unavailable"
    assert failed["prompt_evidence"]["normalized_mse"] is None
    assert failed["prompt_evidence"]["per_target"] == {}
    assert (
        failed["prompt_evidence"]["unavailable_reason"] == "numerical_active_wall_limit"
    )
    assert failed["endpoint"]["parameters"]


def test_heldout_changes_cannot_change_training_feedback(originals, tmp_path):
    changed = _build(tmp_path / "changed", poison=-123456789.0)
    first = evidence.load_training_evidence(*originals[:2])
    second = evidence.load_training_evidence(*changed[:2])
    assert first["training"] == second["training"]
    assert first["training_summary"] == second["training_summary"]
    assert list(first["parents"]) == list(second["parents"])
    for name in first["parents"]:
        for key in ("problem", "endpoint", "prompt_evidence"):
            assert first["parents"][name][key] == second["parents"][name][key]


@pytest.mark.parametrize(
    "field",
    [
        "parameters",
        "parameters_sha256",
        "training_sha256",
        "policy",
        "score",
        "predictions",
        "rows",
        "solver_difference",
    ],
)
def test_resealed_numerical_receipt_changes_are_refused(paths, field):
    source, supplemental, _, _ = paths
    path = supplemental / "results/model_0/operations/output_verification/result.json"
    operation = read_seal(path)
    value = operation["value"]
    if field == "parameters":
        value["parameters"]["theta"] += 0.1
    elif field.endswith("sha256"):
        value[field] = "0" * 64
    elif field == "policy":
        value["policy"]["rtol"] *= 10
    elif field == "score":
        value["normalized_mse"] += 1
    elif field == "predictions":
        value["predictions"]["training_0"]["y"][0] += 1
    elif field == "rows":
        value["predictions"].pop("training_0")
    else:
        value[field] = 0.01
    reseal(path, operation)
    with pytest.raises(ValueError):
        evidence.load_training_evidence(source, supplemental)


def test_failed_verification_cannot_supply_scores(paths):
    source, supplemental, _, _ = paths
    path = supplemental / "results/model_7/operations/output_verification/result.json"
    operation = read_seal(path)
    operation["value"]["normalized_mse"] = 1.0
    reseal(path, operation)
    with pytest.raises(ValueError, match="unavailable verification"):
        evidence.load_training_evidence(source, supplemental)


def test_original_bytes_and_checkpoint_hashes_remain_binding(paths):
    source, supplemental, _, _ = paths
    path = source / "inputs.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="original source bytes"):
        evidence.load_training_evidence(source, supplemental)
    path.write_text(path.read_text()[:-1])
    path = supplemental / "results/model_0/operations/output_verification/progress.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint differs"):
        evidence.load_training_evidence(source, supplemental)


def test_unverified_public_prompt_is_rejected(paths, tmp_path):
    changed = tmp_path / "changed.txt"
    changed.write_text("an unaudited task")
    with pytest.raises(ValueError, match="public prompt identity"):
        evidence.load_training_evidence(*paths[:2], public_prompt=changed)


def _mock_archive(monkeypatch, source, *, corrupt_record=False, alias_mutation=None):
    data = read_seal(source / "inputs.json")
    records = data["records"]
    calls = []
    plan = {
        "artifact_sha256": "a" * 64,
        "cases": [{"case_id": name} for name in records],
    }
    monkeypatch.setattr(
        evidence.inputs,
        "_returned_plan",
        lambda root: (plan, {name: {} for name in records}),
    )
    brief = {
        "scientific_context": "Recover the public response.",
        "public_variables": [{"name": "y", "data_role": "target"}],
        "requirements": [
            {
                "id": "response",
                "public_requirement": "Causal response",
                "targets": ["y"],
                "drivers": ["y"],
                "requires_dynamic_memory": True,
            }
        ],
        "target_dependencies": [],
    }

    def record(root, plan, entry, expected):
        name = entry["case_id"]
        calls.append(name)
        row = deepcopy(records[name])
        if corrupt_record:
            row["source_packet_sha256"] = "0" * 64
        source_value = {
            "brief": brief,
            "public_cell": {
                "target_contract": {
                    "benchmark_id": data["source"]["benchmark_id"],
                    "public_prompt_sha256": data["source"]["public_files"][
                        "proposer_prompt.txt"
                    ],
                }
            },
            "source_input": {
                "input": {
                    "training_evidence": {
                        "split": "train",
                        "trajectories": [
                            {
                                "trajectory_id": r["trajectory_id"],
                                "time": r["time"],
                                "targets": r["targets"],
                                "auxiliaries": r["auxiliaries"],
                                "inputs": r["external_inputs"],
                                "covariates": r["fixed_covariates"],
                            }
                            for r in data["training"]["rows"]
                        ],
                    }
                }
            },
        }
        return row, {"source": source_value}

    monkeypatch.setattr(evidence.inputs, "_record", record)

    def handoff(path):
        row = records[path.parent.name]
        aliases = {"y": "x", "z": "z", "r": "r"}
        value = {
            "artifact_sha256": row["source_packet_sha256"],
            "prefix_fit_problem": row["problem"],
            "gain_lowering": {"state_aliases": aliases},
            "initialization_dependency_contract": {"state_aliases": dict(aliases)},
        }
        if alias_mutation is not None:
            alias_mutation(value)
        return value

    monkeypatch.setattr(evidence.inputs, "sealed_read", handoff)
    return calls


def test_public_briefs_require_archive_chain_and_all_matching_parents(
    paths, monkeypatch
):
    source, supplemental, _, archive = paths
    calls = _mock_archive(monkeypatch, source)
    result = evidence.load_training_evidence(
        source, supplemental, construction_archive=archive
    )
    evidence.validate_training_projection(result)
    assert calls == [f"model_{index}" for index in range(8)]
    for parent in result["parents"].values():
        assert parent["public_brief"]["requirements"][0]["requires_dynamic_memory"]
        assert parent["public_brief_sha256"] == evidence.public.content_sha256(
            parent["public_brief"]
        )
        assert parent["scientific_aliases"] == {"y": "x"}
        assert parent["scientific_aliases_sha256"] == evidence.public.content_sha256(
            {"y": "x"}
        )
    _mock_archive(monkeypatch, source, corrupt_record=True)
    with pytest.raises(ValueError, match="construction record differs"):
        evidence.load_training_evidence(
            source, supplemental, construction_archive=archive
        )


@pytest.mark.parametrize("mutation", ["disagreement", "unknown_state", "identity"])
def test_scientific_aliases_require_matching_authenticated_model(
    paths, monkeypatch, mutation
):
    source, supplemental, _, archive = paths

    def mutate(value):
        if mutation == "identity":
            value["artifact_sha256"] = "0" * 64
        else:
            value["gain_lowering"]["state_aliases"]["y"] = "unknown_state"
            if mutation == "unknown_state":
                value["initialization_dependency_contract"]["state_aliases"]["y"] = (
                    "unknown_state"
                )

    _mock_archive(monkeypatch, source, alias_mutation=mutate)
    with pytest.raises(ValueError, match="scientific alias"):
        evidence.load_training_evidence(
            source, supplemental, construction_archive=archive
        )


@pytest.mark.parametrize(
    "mutation", ["hash", "private_key", "unknown_state", "archive"]
)
def test_portable_projection_checks_scientific_alias_bindings(
    paths, monkeypatch, mutation
):
    source, supplemental, _, archive = paths
    _mock_archive(monkeypatch, source)
    value = evidence.load_training_evidence(
        source, supplemental, construction_archive=archive
    )
    parent = value["parents"]["model_0"]
    if mutation == "hash":
        parent["scientific_aliases_sha256"] = "0" * 64
    elif mutation == "archive":
        value["construction_archive_sha256"] = None
    else:
        parent["scientific_aliases"] = (
            {"z": "z"} if mutation == "private_key" else {"y": "unknown_state"}
        )
        parent["scientific_aliases_sha256"] = evidence.public.content_sha256(
            parent["scientific_aliases"]
        )
    with pytest.raises(ValueError, match="scientific alias"):
        evidence.validate_training_projection(value)


def test_archive_identity_checked_before_reconstructing_history(
    paths, tmp_path, monkeypatch
):
    source, supplemental, _, _archive = paths
    changed = tmp_path / "changed.tar.gz"
    with tarfile.open(changed, "w:gz") as stream:
        member = tarfile.TarInfo("other.txt")
        member.size = 3
        stream.addfile(member, io.BytesIO(b"bad"))
    monkeypatch.setattr(
        evidence.inputs,
        "_returned_plan",
        lambda *a: pytest.fail("wrong archive history opened"),
    )
    with pytest.raises(ValueError, match="archive identity"):
        evidence.load_training_evidence(
            source, supplemental, construction_archive=changed
        )


@pytest.mark.parametrize("where", ["feedback", "target", "witness", "parent", "top"])
def test_portable_projection_rejects_hidden_heldout_fields(originals, where):
    value = evidence.load_training_evidence(*originals[:2])
    parent = value["parents"]["model_0"]
    feedback = parent["prompt_evidence"]
    target = feedback["per_target"]["y"]
    destination = {
        "feedback": feedback,
        "target": target,
        "witness": target["witnesses"][0],
        "parent": parent,
        "top": value,
    }[where]
    destination["heldout_nmse"] = 0.123
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        evidence.validate_training_projection(value)


@pytest.mark.parametrize(
    "mutation",
    ["training", "model", "parameters", "initials", "witness", "status", "budget"],
)
def test_portable_projection_checks_internal_training_bindings(originals, mutation):
    value = evidence.load_training_evidence(*originals[:2])
    parent = value["parents"]["model_0"]
    if mutation == "training":
        value["training"]["rows"][0]["targets"]["y"][0] += 1
    elif mutation == "model":
        parent["problem"]["parameter_guesses"]["theta"] += 0.1
    elif mutation == "parameters":
        parent["endpoint"]["parameters"]["theta"] += 0.1
    elif mutation == "initials":
        value["training_summary"]["initial_observations"][0]["targets"]["y"] += 1
    elif mutation == "witness":
        parent["prompt_evidence"]["per_target"]["y"]["witnesses"][0]["time"] += 1
    elif mutation == "status":
        parent["endpoint"]["verification_status"] = "unavailable"
    else:
        parent["prompt_evidence"]["search"]["budget_exhausted"] = False
    with pytest.raises(ValueError):
        evidence.validate_training_projection(value)


def test_portable_projection_validation_does_not_reopen_source(originals, monkeypatch):
    value = evidence.load_training_evidence(*originals[:2])
    for module, name in (
        (evidence.campaign, "verify"),
        (evidence.campaign, "_source"),
        (evidence.inputs, "_returned_plan"),
        (Path, "read_bytes"),
        (Path, "read_text"),
    ):
        monkeypatch.setattr(
            module, name, lambda *a, **k: pytest.fail("source reopened")
        )
    evidence.validate_training_projection(value)
