"""Independent exact checks for the bounded Gaussian regularity example."""

from fractions import Fraction

import pytest

from scripts import check_premise_exploration_contract as checks

Q = Fraction


@pytest.mark.parametrize("denominator", [100, 101, 1000, 10000, 10**8, 10**30])
def test_boundary_divergences_match_distances_to_alternative_sets(denominator):
    radius = Q(1, denominator)
    result = checks.regularity_boundary(radius)
    answer, mean = Q(result["strictly_correct_answer"]), Q(1, 20)
    tolerance = Q(1, 10)
    # Infima use the nearest boundary even though the alternatives are open.
    left, right = answer - tolerance, answer + tolerance
    expected_singleton = min((mean - left) ** 2, (right - mean) ** 2) / 2
    expected_common = (answer + radius + tolerance - mean) ** 2 / 2
    assert Q(result["singleton_divergence"]) == expected_singleton
    assert Q(result["neighborhood_divergence"]) == expected_common
    assert expected_common - expected_singleton > Q(1, 100)
    assert abs(mean - answer) < tolerance


@pytest.mark.parametrize("radius", [Q(0), Q(-1, 100), Q(1, 50), 0.001, "1/100"])
def test_outside_proved_radius_branch_is_rejected(radius):
    with pytest.raises(ValueError, match="exact Fraction"):
        checks.regularity_boundary(radius)


def test_disconnected_retained_set_loses_its_adequate_center():
    assert checks.retained_radius((Q(0), Q(1), Q(2))) == 1
    assert checks.retained_radius((Q(0), Q(2))) == 2
    assert max(abs(Q(1) - point) for point in (Q(0), Q(2))) == 1


@pytest.mark.parametrize("values", [(), (0, 1), (Q(1), 2.0)])
def test_invalid_retained_set_is_rejected(values):
    with pytest.raises(ValueError, match="nonempty"):
        checks.retained_radius(values)


def test_receipt_reuse_and_changed_result_rejection(tmp_path):
    path = tmp_path / "contract-check.json"
    result = checks.report()
    assert checks.save_receipt(path, result) == "written"
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert checks.save_receipt(path, result) == "reused"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    result["protocol"] = "altered"
    with pytest.raises(ValueError, match="Existing receipt differs"):
        checks.save_receipt(path, result)
    assert path.read_bytes() == before[0]
