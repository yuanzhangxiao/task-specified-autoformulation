"""Response timing, ambiguity, full-grid identity and balanced prompt packing."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.data import SplitName
from autoformalism.fitting import public_fitting as public
from autoformalism.search.residual_evidence import build_residual_evidence
from autoformalism.search.response_evidence import (
    build_response_evidence,
    compact_user,
    presentation,
    shape,
)
from autoformalism.search.response_revision import payload
from scripts.smoke_review_multi import example
from tests.test_residual_evidence import CONTEXT, fixture, packet


def test_nonuniform_peak_time_half_return_and_signed_excursion():
    time = np.array([0.0, 1.0, 3.0, 6.0, 9.0, 12.0])
    for sign in (1, -1):
        value = shape(time, 10 + sign * np.array([0.0, 1.0, 4.0, 3.0, 1.0, 0.0]), 0.01)
        assert value.peak_status == "interior" and value.peak_time == 3
        assert value.excursion == sign * 4
        assert value.half_return_elapsed == 6
        assert value.half_return_status == "observed"


@pytest.mark.parametrize(
    "values,status,decay",
    [
        ([0, 0, 0, 0], "flat", "undefined"),
        ([0, 1, 2, 3], "boundary", "undefined"),
        ([0, 3, 3, 0], "plateau", "undefined"),
        ([0, 3, 2.5, 2], "interior", "not_reached"),
    ],
)
def test_ambiguous_or_censored_timing_is_not_a_zero_delay(values, status, decay):
    result = shape(np.arange(4.0), np.array(values, dtype=float), 0.01)
    assert result.peak_status == status and result.half_return_status == decay
    assert result.half_return_elapsed is None
    if status != "interior":
        assert result.peak_time is None


def test_full_grid_single_input_delay_and_repeated_event_ambiguity():
    data, _, _ = fixture(count=1)
    row = replace(
        data.trajectories[0],
        targets={"y": np.array([0.0, 0.0, 2.0, 4.0, 2.0, 1.0, 0.0])},
        external_inputs={"u": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
    )
    data = replace(data, trajectories=(row,))
    current = {
        row.trajectory_id: {
            "time": row.time.tolist(),
            "predictions": {"y": [0.0, 0.0, 1.0, 2.0, 4.0, 2.0, 0.0]},
        }
    }
    saved = packet(data, current).model_dump(mode="json")
    result = build_response_evidence(data, CONTEXT, current, saved)
    assert result.rows[0].input_peak_delays == {"u": (2.0, 3.0)}
    assert result.rows[0].peak_timing_error == 1
    row = replace(
        row, external_inputs={"u": np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
    )
    data = replace(data, trajectories=(row,))
    result = build_response_evidence(
        data, CONTEXT, current, packet(data, current).model_dump(mode="json")
    )
    assert result.inputs[row.trajectory_id]["u"].departure_count == 2
    assert result.rows[0].input_peak_delays == {}


def test_train_only_alignment_and_replay_identity():
    data, current, _ = fixture()
    saved = packet(data, current).model_dump(mode="json")
    result = build_response_evidence(data, CONTEXT, current, saved)
    assert len(result.rows) == 4
    assert "hidden" not in result.numerical_status
    with pytest.raises(ValueError, match="train split"):
        build_response_evidence(
            replace(data, name=SplitName.VALIDATION), CONTEXT, current, saved
        )
    changed = copy.deepcopy(current)
    changed["train_0"]["predictions"]["y"][1] += 1
    with pytest.raises(ValueError, match="predictions differ"):
        build_response_evidence(data, CONTEXT, changed, saved)
    changed = copy.deepcopy(saved)
    changed["training_content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="training identity"):
        build_response_evidence(data, CONTEXT, current, changed)


def response_example():
    bundle, saved, params, request, train, _ = example()
    current = {
        r.trajectory_id: {"time": r.time, "predictions": r.targets} for r in train.rows
    }
    full = build_response_evidence(
        public.unpack_split(train), request.context, current, saved
    )
    return bundle, saved, params, full.model_dump(mode="json")


def test_target_balanced_packing_preserves_contract_and_cites_only_shown_rows():
    bundle, saved, params, full = response_example()
    user = payload(bundle, saved, params, full)
    user["public_target_contract"] = {"required": ["Gp", "I", "U"]}
    before = copy.deepcopy(user)
    reduced = compact_user(user, per_target=1, samples=False)
    assert user == before
    evidence = reduced["training_evidence"]
    assert set(evidence["target_overview"]) == {"Gp", "I", "U"}
    assert len(evidence["examples"]) == 3 and evidence["diagnostic_samples"] == []
    assert evidence["omitted_trajectory_target_examples"] == 3
    assert {r["evidence_id"] for r in evidence["examples"]} == {
        r["ref"] for r in reduced["evidence_catalog"]
    }
    for key in (
        "model",
        "public_target_contract",
        "initialization_plan",
        "retained_fitted_parameters",
    ):
        assert reduced[key] == before[key]
    wrong = copy.deepcopy(saved)
    wrong["normalized_mse"] += 0.1
    with pytest.raises(ValueError, match="source differ"):
        presentation(full, wrong)


def test_short_refs_do_not_verify_omitted_rows():
    from autoformalism.search.review_revision_multi import apply_edits

    bundle, saved, params, full = response_example()
    user = compact_user(
        payload(bundle, saved, params, full), per_target=1, samples=False
    )
    refs = {r["ref"]: r["record"] for r in user["evidence_catalog"]}
    kept = next(iter(refs))
    result = apply_edits(
        bundle,
        saved,
        {
            "hypothesis": "A baseline offset may matter.",
            "evidence_refs": [kept, "R002", "E001"],
            "output_mappings": [{"channel": "U", "expression": "disposal+offset"}],
            "new_parameters": [{"name": "offset"}],
        },
        reference_catalog=refs,
    )
    audit = result["provenance"]["citation_audit"]
    assert audit["valid_evidence_ids"] == [refs[kept]]
    assert audit["unresolved_references"] == ["R002", "E001"]


def test_sixteen_trajectory_three_target_packet_has_nine_examples():
    _, _, params, request, train, _ = example()
    original = public.unpack_split(train)
    data = replace(
        original,
        trajectories=tuple(
            replace(original.trajectories[i % 2], trajectory_id=f"train_{i:03d}")
            for i in range(16)
        ),
    )
    current = {
        t.trajectory_id: {
            "time": t.time.tolist(),
            "predictions": {
                k: (v + (i % 4) * 0.1).tolist() for k, v in t.targets.items()
            },
        }
        for i, t in enumerate(data.trajectories)
    }
    saved = build_residual_evidence(
        data,
        request.context,
        current,
        candidate_sha256="a" * 64,
        parameters=params,
        numerical_status={},
    ).model_dump(mode="json")
    full = build_response_evidence(data, request.context, current, saved).model_dump(
        mode="json"
    )
    shown, refs = presentation(full, saved)
    assert len(saved["rows"]) == 48 and len(shown["examples"]) == len(refs) == 9
    assert len(shown["target_overview"]) == 3
    assert shown["omitted_trajectory_target_examples"] == 39
    for target in request.context.targets:
        rows = [r for r in saved["rows"] if r["target"] == target]
        n = sum(r["sample_count"] for r in rows)
        expected = sum(r["sample_count"] * r["normalized_mse"] for r in rows) / n
        assert shown["target_overview"][target]["nmse"] == pytest.approx(
            expected, rel=1e-5
        )
