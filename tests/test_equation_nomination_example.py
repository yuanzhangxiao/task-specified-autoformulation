"""The example is selected before inspecting held forecasts."""

from scripts.plot_equation_nomination_example import nominated_target


def test_example_requires_fully_measured_rate_and_selected_family():
    row = {
        "status": "all_arguments_observed",
        "response": "interval_average_rate",
        "shortlist": [{"shape": "hinge x"}],
    }
    packet = {
        "shape": {
            "candidate_families": [{"z": "hinge x"}, {"z": "affine"}],
            "evidence": {"equations": {"z": row}},
        }
    }
    assert nominated_target(packet, 0) == "z"
    assert nominated_target(packet, 1) is None
    assert nominated_target(packet, None) is None
    row["status"] = "observed_projection"
    assert nominated_target(packet, 0) is None
