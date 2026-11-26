import pytest

from scripts.audit_t2_disposal_readout import inspect_readout


def event(index: int, insulin: float, effective: float) -> dict:
    return {
        "trajectory_id": f"train_{index}",
        "time": [0, 1],
        "targets": {"I": [insulin, insulin + 1], "U": [2 + effective, 5]},
        "auxiliaries": {"Uii": [2, 2]},
        "inputs": {},
        "covariates": {},
    }


def test_calibrated_output_and_shared_function_error_bound():
    rows = [event(0, 1, 4), event(1, 1, 2), event(2, 2, 8)]
    result = inspect_readout(rows)
    assert result["samples"] == 6 and result["sample_range"] == [2, 8]
    witness = result["largest_repeated_input_witness"]
    assert witness["disposal_at_start"] == [2, 4]
    assert witness["minimum_uniform_error_at_start"] == 1
    assert inspect_readout(rows[1:])["largest_repeated_input_witness"] is None


def test_disallows_official_validation_events():
    row = event(0, 1, 0)
    row["trajectory_id"] = "validation_0"
    with pytest.raises(ValueError, match="train"):
        inspect_readout([row])
