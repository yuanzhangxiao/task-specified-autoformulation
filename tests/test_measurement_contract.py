"""Known-noise provenance, fixed masks, explicit initials and identity controls."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.schemas.measurement_fitting import (
    MeasurementProblem,
    analysis_layout,
    analysis_training,
    initial_values,
)
from autoformalism.schemas.public_fitting import PublicFitRequest


def measurement_payload(*, missing=False, shared=False):
    """Small raw-unit, exact-clock public control with optional uncertain initial."""
    source_hash = "a" * 64
    preparation_hash = "b" * 64
    names = ["rate", "initial"] if shared else ["rate"]
    readings = []
    for index, clock in enumerate((10, 12, 14)):
        for channel, raw in (("y", 1 + index), ("u", 2)):
            is_missing = missing and channel == "y" and index == 1
            readings.append(
                {
                    "id": f"r_{channel}_{index}",
                    "artifact_sha256": source_hash,
                    "source_row": index,
                    "channel": channel,
                    "clock_value": clock,
                    "raw_value": None if is_missing else raw,
                    "condition": "missing" if is_missing else "present",
                }
            )
    present = [
        r["id"] for r in readings if r["channel"] == "y" and r["condition"] == "present"
    ]
    assumptions = [
        {
            "id": topic,
            "topic": topic,
            "status": "known",
            "sources": ["preparation" if topic == "initial" else "acquisition"],
            "reason": f"Public protocol fixes {topic}; conditional on that protocol.",
        }
        for topic in (
            "noise",
            "independence",
            "time",
            "input",
            "calibration",
            "initial",
        )
    ]
    return {
        "request": {
            "base_candidate": {
                "candidate_id": "measurement_control",
                "parent_candidate_id": None,
                "states": [{"name": "x", "kind": "latent"}],
                "state_equations": [{"state": "x", "rhs": "-rate*x+u"}],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "parameters": [{"name": name, "scope": "global"} for name in names],
                "initial_conditions": [
                    {"state": "x", "scope": "global", "fixed_value": 0}
                ],
            },
            "context": {"targets": ["y"], "external_inputs": ["u"]},
            "initialization_plan": {"rules": {}},
            "parameter_guesses": dict.fromkeys(names, 1.0),
            "profile": "general-rollout-v1",
            "source": {
                "stage": "synthetic_control",
                "task_id": "measurement_control",
                "artifact_sha256": source_hash,
            },
        },
        "manifest": {
            "sources": [
                {
                    "id": "acquisition",
                    "artifact_sha256": source_hash,
                    "kind": "acquisition",
                },
                {
                    "id": "preparation",
                    "artifact_sha256": preparation_hash,
                    "kind": "public_preparation",
                },
            ],
            "assumptions": assumptions,
            "artifacts": [
                {
                    "sha256": source_hash,
                    "media_type": "application/json",
                    "acquisition_source": "acquisition",
                    "parser_protocol": "public_json_v1",
                }
            ],
            "channels": [
                {
                    "id": "y",
                    "role": "target",
                    "raw_unit": "raw_reading",
                    "analysis_unit": "state_unit",
                    "gain": 2,
                    "offset": 1,
                    "calibration_assumption": "calibration",
                },
                {
                    "id": "u",
                    "role": "external_input",
                    "raw_unit": "state_unit_per_second",
                    "analysis_unit": "state_unit_per_second",
                    "gain": 1,
                    "offset": 0,
                    "calibration_assumption": "calibration",
                    "exact_input_assumption": "input",
                },
            ],
            "time_bases": [
                {
                    "trajectory": "train_a",
                    "raw_unit": "ticks",
                    "analysis_unit": "seconds",
                    "origin": 10,
                    "factor": 0.5,
                    "assumption": "time",
                }
            ],
            "forcing_rules": [
                {
                    "trajectory": "train_a",
                    "channel": "u",
                    "knot_reading_ids": ["r_u_0", "r_u_1", "r_u_2"],
                    "assumption": "input",
                }
            ],
            "noise": {
                "reading_ids": present,
                "sigma": [0.2] * len(present),
                "scale_assumption": "noise",
                "independence_assumption": "independence",
            },
        },
        "training": {
            "rows": [
                {
                    "trajectory_id": "train_a",
                    "clock_values": [10, 12, 14],
                    "readings": readings,
                }
            ]
        },
        "initial_rules": [
            {
                "state": "x",
                "mode": "shared_value" if shared else "exact_boundary",
                "exact_boundaries": []
                if shared
                else [
                    {
                        "trajectory": "train_a",
                        "value": 4,
                        "preparation_source": "preparation",
                    }
                ],
                "shared_coordinate": "initial" if shared else None,
                "assumption": "initial",
            }
        ],
        "coordinates": {
            "parameters": {name: {"center": 1, "scale": 1} for name in names},
            "states": {"x": {"center": 0, "scale": 1}},
            "provenance": {},
        },
        "start": dict.fromkeys(names, 1.0),
        "incumbent": dict.fromkeys(names, 1.0),
    }


def test_conversion_mask_and_sigma_do_not_modify_raw_data():
    payload = measurement_payload(missing=True)
    before = deepcopy(payload)
    problem = MeasurementProblem.model_validate(payload)
    layout = analysis_layout(problem)
    assert layout.values == (3, None, 7)
    assert layout.analysis_times == (0, 1, 2)
    assert layout.sigma == (0.2, None, 0.2)
    assert layout.included == (True, False, True)
    numerical = analysis_training(problem)
    assert numerical.rows[0].targets["y"] == (3, 0, 7)
    assert initial_values(problem, "train_a", problem.start) == {"x": 4}
    assert payload == before
    assert problem.training.rows[0].readings[2].raw_value is None
    assert MeasurementProblem.model_validate_json(problem.model_dump_json()) == problem


def test_shared_initial_parameter_is_preserved_without_using_first_reading():
    problem = MeasurementProblem.model_validate(measurement_payload(shared=True))
    assert {p.name for p in problem.request.base_candidate.parameters} == {
        "rate",
        "initial",
    }
    assert initial_values(problem, "train_a", {"rate": 1, "initial": -5}) == {"x": -5}


def test_nested_mappings_and_raw_records_are_immutable_and_detached():
    payload = measurement_payload()
    request = PublicFitRequest.model_validate(payload["request"])
    payload["request"] = request
    problem = MeasurementProblem.model_validate(payload)
    request.parameter_guesses["rate"] = 9
    assert problem.request.parameter_guesses["rate"] == 1
    for mapping in (
        problem.start,
        problem.incumbent,
        problem.request.parameter_guesses,
        problem.request.context.forcing_bounds,
        problem.coordinates.parameters,
        problem.coordinates.provenance,
    ):
        with pytest.raises(TypeError, match="immutable"):
            mapping["new"] = 1
    with pytest.raises(ValidationError):
        problem.training.rows[0].readings[0].raw_value = 99


def test_wrapper_preserves_legacy_request_hash_exactly():
    request = PublicFitRequest.model_validate(measurement_payload()["request"])
    digest = content_sha256(request)
    payload = measurement_payload()
    payload["request"] = request
    problem = MeasurementProblem.model_validate(payload)
    assert content_sha256(problem.request) == digest
    assert problem.request.model_dump(mode="json") == request.model_dump(mode="json")
    assert problem.request.protocol == "public-fit-1"


@pytest.mark.parametrize(
    "change",
    [
        "sigma",
        "calibration",
        "clock",
        "source",
        "parser",
        "initial",
        "start",
        "coordinate",
        "raw",
        "mask",
    ],
)
def test_every_objective_or_provenance_change_changes_identity(change):
    payload = measurement_payload()
    before = content_sha256(MeasurementProblem.model_validate(payload))
    if change == "sigma":
        payload["manifest"]["noise"]["sigma"][0] = 0.3
    elif change == "calibration":
        payload["manifest"]["channels"][0]["gain"] = -2
    elif change == "clock":
        payload["manifest"]["time_bases"][0]["factor"] = 1
    elif change == "source":
        payload["manifest"]["sources"][1]["artifact_sha256"] = "c" * 64
    elif change == "parser":
        payload["manifest"]["artifacts"][0]["parser_protocol"] = "public_json_v2"
    elif change == "initial":
        payload["initial_rules"][0]["exact_boundaries"][0]["value"] = 5
    elif change == "start":
        payload["start"]["rate"] = 2
    elif change == "coordinate":
        payload["coordinates"]["parameters"]["rate"]["scale"] = 2
    elif change == "raw":
        payload["training"]["rows"][0]["readings"][0]["raw_value"] = 3
    else:
        payload = measurement_payload(missing=True)
    assert content_sha256(MeasurementProblem.model_validate(payload)) != before


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf")])
def test_invalid_sigma_never_becomes_an_epsilon(sigma):
    payload = measurement_payload()
    payload["manifest"]["noise"]["sigma"][0] = sigma
    with pytest.raises(ValidationError):
        MeasurementProblem.model_validate(payload)


@pytest.mark.parametrize(
    "topic", ["noise", "independence", "time", "input", "calibration", "initial"]
)
@pytest.mark.parametrize("status", ["estimated", "unavailable"])
def test_uncertain_assumptions_are_explicitly_unsupported(topic, status):
    payload = measurement_payload()
    next(a for a in payload["manifest"]["assumptions"] if a["topic"] == topic)[
        "status"
    ] = status
    with pytest.raises(ValidationError, match=f"{topic.upper()}_UNSUPPORTED"):
        MeasurementProblem.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        "val",
        "private",
        "censor",
        "covariance",
        "nuisance",
        "map",
        "no_unit",
        "zero_gain",
        "noise_order",
        "missing_not_null",
        "missing_forcing",
        "duplicate_id",
        "duplicate_sample",
        "bad_source",
        "bad_initial_source",
        "bad_boundary",
        "incomplete_coordinates",
        "legacy_initials",
        "no_target",
        "nonlinear_forcing",
        "wrong_grid",
    ],
)
def test_closed_contract_rejects_malformed_or_unsupported_capabilities(change):
    payload = measurement_payload()
    manifest = payload["manifest"]
    row = payload["training"]["rows"][0]
    if change == "val":
        payload["training"]["split"] = "val"
    elif change == "private":
        payload["reference_parameters"] = {"rate": 1}
    elif change == "censor":
        row["readings"][0]["condition"] = "left_censored"
    elif change == "covariance":
        manifest["noise"]["covariance"] = [[1]]
    elif change == "nuisance":
        payload["observation_nuisances"] = ["gain"]
    elif change == "map":
        payload["initial_rules"][0]["mode"] = "shared_exact_causal_map"
    elif change == "no_unit":
        del manifest["channels"][0]["raw_unit"]
    elif change == "zero_gain":
        manifest["channels"][0]["gain"] = 0
    elif change == "noise_order":
        manifest["noise"]["reading_ids"].reverse()
    elif change == "missing_not_null":
        row["readings"][0]["condition"] = "missing"
    elif change == "missing_forcing":
        row["readings"][1].update(condition="missing", raw_value=None)
    elif change == "duplicate_id":
        row["readings"][1]["id"] = row["readings"][0]["id"]
    elif change == "duplicate_sample":
        duplicate = deepcopy(row["readings"][0])
        duplicate.update(id="other", source_row=10)
        row["readings"].append(duplicate)
    elif change == "bad_source":
        manifest["assumptions"][0]["sources"] = ["unknown"]
    elif change == "bad_initial_source":
        payload["initial_rules"][0]["exact_boundaries"][0]["preparation_source"] = (
            "acquisition"
        )
    elif change == "bad_boundary":
        payload["initial_rules"][0]["exact_boundaries"][0]["trajectory"] = "unknown"
    elif change == "incomplete_coordinates":
        payload["start"] = {}
    elif change == "legacy_initials":
        payload["request"]["initialization_plan"]["rules"] = {
            "x": {"initial": {"mode": "value", "guess": 0}}
        }
    elif change == "no_target":
        for reading in row["readings"]:
            if reading["channel"] == "y":
                reading.update(condition="missing", raw_value=None)
    elif change == "nonlinear_forcing":
        manifest["forcing_rules"][0]["interpolation"] = "left_hold"
    elif change == "wrong_grid":
        row["readings"][0]["clock_value"] = 11
    with pytest.raises(ValidationError):
        MeasurementProblem.model_validate(payload)


def test_missing_first_reading_is_not_used_as_zero_initial():
    payload = measurement_payload()
    payload["training"]["rows"][0]["readings"][0].update(
        raw_value=None, condition="missing"
    )
    payload["manifest"]["noise"]["reading_ids"].pop(0)
    payload["manifest"]["noise"]["sigma"].pop(0)
    problem = MeasurementProblem.model_validate(payload)
    assert analysis_training(problem).rows[0].targets["y"][0] == 0
    assert analysis_layout(problem).included == (False, True, True)
    assert initial_values(problem, "train_a", problem.start) == {"x": 4}
    assert problem.training.rows[0].readings[0].raw_value is None


def test_distinct_exact_preparations_and_multichannel_layout_are_preserved():
    payload = measurement_payload()
    request = payload["request"]
    request["context"]["targets"].append("y2")
    request["base_candidate"]["observation_mappings"].append(
        {"channel": "y2", "expression": "2*x"}
    )
    channel = deepcopy(payload["manifest"]["channels"][0])
    channel.update(id="y2", gain=1, offset=0)
    payload["manifest"]["channels"].append(channel)
    first = payload["training"]["rows"][0]
    for reading in tuple(first["readings"]):
        if reading["channel"] == "y":
            second_channel = deepcopy(reading)
            second_channel.update(id=reading["id"] + "_second", channel="y2")
            first["readings"].append(second_channel)
    second = deepcopy(first)
    second["trajectory_id"] = "train_b"
    for reading in second["readings"]:
        reading["id"] += "_b"
        reading["source_row"] += 3
    payload["training"]["rows"].append(second)
    clock = deepcopy(payload["manifest"]["time_bases"][0])
    clock["trajectory"] = "train_b"
    payload["manifest"]["time_bases"].append(clock)
    forcing = deepcopy(payload["manifest"]["forcing_rules"][0])
    forcing.update(
        trajectory="train_b", knot_reading_ids=[f"r_u_{i}_b" for i in range(3)]
    )
    payload["manifest"]["forcing_rules"].append(forcing)
    payload["initial_rules"][0]["exact_boundaries"].append(
        {"trajectory": "train_b", "value": 9, "preparation_source": "preparation"}
    )
    ids = [
        f"r_y_{i}{suffix}{trajectory_suffix}"
        for trajectory_suffix in ("", "_b")
        for i in range(3)
        for suffix in ("", "_second")
    ]
    payload["manifest"]["noise"].update(reading_ids=ids, sigma=[0.2] * 12)
    problem = MeasurementProblem.model_validate(payload)
    assert analysis_layout(problem).reading_ids == tuple(ids)
    assert analysis_layout(problem, ["y2", "y"]).values[:4] == (1, 3, 2, 5)
    assert initial_values(problem, "train_b", problem.start) == {"x": 9}
    assert initial_values(problem, "train_a", problem.start) == {"x": 4}
    with pytest.raises(ValueError, match="exactly the declared"):
        analysis_layout(problem, ["y"])


def test_fixed_covariates_are_exact_single_public_records():
    payload = measurement_payload()
    payload["request"]["context"]["fixed_covariates"] = ["area"]
    payload["manifest"]["channels"].append(
        {
            "id": "area",
            "role": "fixed_covariate",
            "raw_unit": "square_cm",
            "analysis_unit": "square_m",
            "gain": 0.0001,
            "offset": 0,
            "calibration_assumption": "calibration",
            "exact_input_assumption": "input",
        }
    )
    payload["training"]["rows"][0]["readings"].append(
        {
            "id": "area_value",
            "artifact_sha256": "a" * 64,
            "source_row": 0,
            "channel": "area",
            "clock_value": 10,
            "raw_value": 20000,
            "condition": "present",
        }
    )
    problem = MeasurementProblem.model_validate(payload)
    assert analysis_training(problem).rows[0].fixed_covariates == {"area": 2}
    payload["manifest"]["channels"][-1]["exact_input_assumption"] = None
    with pytest.raises(ValidationError, match="INPUT_UNSUPPORTED"):
        MeasurementProblem.model_validate(payload)
