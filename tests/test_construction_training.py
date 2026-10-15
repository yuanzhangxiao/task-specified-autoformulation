"""Public provenance, concise observations and qualified advisory questions."""

import json
from pathlib import Path

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.research.construction_training import (
    compatibility_feedback,
    export_bundle,
    initial_highlights,
    load_bundle,
    load_evidence,
)
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    EvidenceContract,
    InitialRule,
    build_evidence,
)
from autoformalism.search.training_evidence import build_training_evidence


def fixture():
    context = ValidationContext(targets=("a",), external_inputs=("u",))
    trajectories = tuple(
        Trajectory(
            trajectory_id=f"trajectory{i}",
            time=np.asarray((0, 1, 2), float),
            targets={"a": np.asarray(values, float)},
            auxiliaries={},
            external_inputs={"u": np.zeros(3)},
            fixed_covariates={},
            derivatives={},
        )
        for i, values in enumerate(((0, 1, 0), (0, 3, 0)))
    )
    train = DatasetSplit(SplitName.TRAIN, trajectories, "fixture")
    contract = EvidenceContract(
        provenance="Explicit fixture runtime assumptions; not inferred from samples",
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=True,
        uniqueness_assumed=True,
        interpolation={"u": "linear_complete_knots"},
        absolute_error_bounds={"a": 0, "u": 0},
        initial_rules={"a": InitialRule(mode="observed_initial")},
    )
    draft = Draft.model_validate(
        {
            "variables": [
                {"name": "a", "definition": "differential", "scientific_role": "output"}
            ],
            "equations": [
                {
                    "name": "a",
                    "terms": [
                        {
                            "sources": ["a", "u"],
                            "scientific_role": "response",
                            "outer_weight_sign": "positive",
                        }
                    ],
                }
            ],
        }
    )
    return context, train, contract, draft


def test_highlights_are_bounded_factual_deterministic_and_show_contrast():
    context, train, contract, _ = fixture()
    evidence = build_evidence(train, context, contract)
    text = initial_highlights(evidence, maximum_characters=1000)
    assert len(text) <= 1000
    assert text == initial_highlights(evidence, maximum_characters=1000)
    assert "same" not in text or "causal" in text
    assert "identical recorded input samples/covariates" in text
    assert "a=1 vs 3" in text
    assert "Other initial states" in text
    assert "omitted=" in text
    assert "not a causal delay" in text
    with pytest.raises(ValueError, match="at least"):
        initial_highlights(evidence, maximum_characters=200)
    full = initial_highlights(evidence)
    assert "constant sampled inputs [u=0], yet a changes" in full
    assert "does not identify a missing dependence" in full


def test_qualified_questions_keep_witnesses_and_never_become_a_gate():
    context, train, contract, draft = fixture()
    evidence = build_evidence(train, context, contract)
    feedback = compatibility_feedback(draft, evidence)
    assert feedback["acceptance_gate"] is False
    questions = feedback["advisory_questions"]
    assert {q["property"] for q in questions} == {
        "scalar_autonomous",
        "closed_subsystem",
    }
    assert all(q["assumptions"] and q["witnesses"] for q in questions)
    dynamic = next(q for q in questions if q["property"] == "closed_subsystem")
    assert dynamic["conditioning"]["matched_initial_information"]["trajectory0"]["a"][
        "recorded_arguments"
    ] == {"a": 0}
    assert dynamic["conditioning"]["matched_complete_forcing_channels"] == ["u"]
    assert all("intended approximation" in q["interpretation"] for q in questions)
    assert feedback == compatibility_feedback(draft, evidence)
    changed = build_evidence(
        train, context, contract.model_copy(update={"uniqueness_assumed": None})
    )
    assert not compatibility_feedback(draft, changed)["advisory_questions"]
    assert compatibility_feedback(draft, changed)["counts"]["deferred"] == 3


def test_evidence_rejects_heldout_forged_digest_and_old_bounded_packet():
    context, train, contract, _ = fixture()
    evidence = build_evidence(train, context, contract)
    assert load_evidence(evidence.model_dump(mode="json")) == evidence
    changed = evidence.model_dump(mode="json")
    changed["trajectories"][0]["inputs"]["u"][0] = 1
    with pytest.raises(ValueError, match="digest"):
        load_evidence(changed)
    for split in ("validation", "test"):
        with pytest.raises(ValueError):
            load_evidence(evidence.model_dump(mode="json") | {"split": split})
    with pytest.raises(ValueError):
        load_evidence(build_training_evidence(train, context).model_dump(mode="json"))


def write_public(path: Path, value: dict):
    path.write_text(json.dumps({"value": value, "sha256": content_sha256(value)}))


def export_fixture(tmp_path, *, name="train", noise=0):
    context, train, contract, _ = fixture()
    case = tmp_path / "public" / "generic_case"
    case.mkdir(parents=True)
    rows = [
        {
            "trajectory_id": item.trajectory_id,
            "time": item.time.tolist(),
            "targets": {k: v.tolist() for k, v in item.targets.items()},
            "external_inputs": {k: v.tolist() for k, v in item.external_inputs.items()},
            "auxiliaries": {},
            "fixed_covariates": {},
        }
        for item in train.trajectories
    ]
    write_public(case / "train.json", {"name": name, "rows": rows})
    write_public(
        case / "specification.json",
        {
            "benchmark_id": "generic_case",
            "noise_sd_fraction": noise,
            "initial_observation_noise": 0,
            "input_contract": "continuous-rates-1",
        },
    )
    # Invalid JSON in held-out files proves the exporter does not inspect them.
    (case / "validation.json").write_text("not valid JSON")
    (case / "test.json").write_text("not valid JSON")
    source = tmp_path / "plan.json"
    sealed_write(
        source,
        {
            "cells": {
                "generic_case": {
                    "context": context.model_dump(mode="json"),
                    "evidence": build_training_evidence(train, context).model_dump(
                        mode="json"
                    ),
                }
            }
        },
    )
    return source, tmp_path / "public", tmp_path / "bundle.json", contract


def test_export_bundle_matches_historical_source_and_resumes(tmp_path):
    args = export_fixture(tmp_path)
    value = export_bundle(*args)
    assert export_bundle(*args) == value
    assert load_bundle(args[2], json.loads(args[0].read_text())) == value
    assert value["test_data_opened"] is False
    evidence = load_evidence(value["cells"]["generic_case"])
    assert evidence.contract.absolute_error_bounds == {"a": 0, "u": 0}
    assert "Explicit fixture runtime assumptions" in evidence.contract.provenance
    case_path = args[1] / "generic_case" / "train.json"
    train = json.loads(case_path.read_text())["value"]
    train["rows"][0]["targets"]["a"][1] = 2
    write_public(case_path, train)
    with pytest.raises(ValueError, match="historical training source"):
        export_bundle(*args)


def test_export_rejects_nontraining_and_digest_before_rows(tmp_path):
    args = export_fixture(tmp_path, name="test")
    with pytest.raises(ValueError, match="train only"):
        export_bundle(*args)
    path = args[1] / "generic_case" / "train.json"
    value = json.loads(path.read_text())
    value["value"]["name"] = "train"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest"):
        export_bundle(*args)


@pytest.mark.parametrize("noise", [None, 0.01])
def test_export_missing_noise_metadata_does_not_assume_exactness(tmp_path, noise):
    args = export_fixture(tmp_path, noise=noise)
    value = export_bundle(*args)
    evidence = load_evidence(value["cells"]["generic_case"])
    assert evidence.contract.absolute_error_bounds == {"a": None, "u": None}
    assert not compatibility_feedback(fixture()[-1], evidence)["advisory_questions"]


def test_changed_candidate_produces_distinct_question_identity():
    context, train, contract, draft = fixture()
    evidence = build_evidence(train, context, contract)
    before = compatibility_feedback(draft, evidence)
    raw = draft.model_dump(mode="json")
    raw["equations"][0]["terms"][0]["sources"] = ["u"]
    after = compatibility_feedback(Draft.model_validate(raw), evidence)
    assert before["topology_sha256"] != after["topology_sha256"]
    assert {q["question_id"] for q in before["advisory_questions"]}.isdisjoint(
        {q["question_id"] for q in after["advisory_questions"]}
    )


def test_export_does_not_invent_runtime_assumptions(tmp_path):
    source, public, output, _ = export_fixture(tmp_path)
    unknown = EvidenceContract(provenance="Runtime hypotheses are unavailable")
    value = export_bundle(source, public, output, unknown)
    evidence = load_evidence(value["cells"]["generic_case"])
    assert evidence.contract.uniqueness_assumed is None
    assert evidence.contract.open_loop is None
    assert evidence.contract.initial_rules == {}
    assert evidence.contract.interpolation == {"u": "unknown"}
    assert not compatibility_feedback(fixture()[-1], evidence)["advisory_questions"]


def test_algebraic_question_displays_actual_complete_argument_values():
    context, train, contract, draft = fixture()
    evidence = build_evidence(train, context, contract)
    raw = draft.model_dump(mode="json")
    raw["variables"][0]["definition"] = "algebraic"
    raw["equations"][0]["terms"][0]["sources"] = ["u"]
    model = Draft.model_validate(raw)
    before = model.model_dump(mode="json")
    questions = compatibility_feedback(model, evidence)["advisory_questions"]
    assert len(questions) == 1
    witnesses = questions[0]["conditioning"]["matched_complete_supplied_arguments"]
    assert len(witnesses) == 2
    assert all(row["values"] == {"u": 0} for row in witnesses)
    assert model.model_dump(mode="json") == before
