"""Check the fixed semantic controls and immutable arithmetic-check receipts."""

from decimal import Context, Decimal, localcontext

import pytest

from scripts import check_premise_revision_controls as checks


def test_bounds_contain_independent_high_precision_evaluations():
    report = checks.controls()
    with localcontext(Context(prec=100)):
        rate = Decimal("0.2")
        feature = ((-rate * 2).exp() - Decimal(-2).exp()) / (1 - rate)
        half_difference = (1 - Decimal("-0.4").exp()) * feature / 2
        point_risk = (Decimal("5.4").sqrt() - 1) / 2 - Decimal("0.1")
    for bounds, expected in (
        (
            report["one_time_product"]["unrestricted_prediction_risk_lower_bound"],
            half_difference,
        ),
        (report["two_query_model_restriction"]["point_model_risk"], point_risk),
    ):
        assert Decimal(bounds[0]) <= expected <= Decimal(bounds[1])
    assert half_difference > Decimal("0.1")
    assert point_risk > Decimal("0.495")


def test_changed_arithmetic_source_is_rejected(monkeypatch):
    monkeypatch.setattr(checks, "INTERVAL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="requalify"):
        checks.controls()


def test_reuse_preserves_receipt_bytes_and_modification_time(tmp_path):
    path = tmp_path / "run" / "receipt.json"
    report = checks.controls()
    assert checks.save_receipt(path, report) == "written"
    original = path.read_bytes(), path.stat().st_mtime_ns
    assert checks.save_receipt(path, report) == "reused"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("existing", [b"partial", b"{}\n"])
def test_incomplete_or_different_receipt_is_not_overwritten(tmp_path, existing):
    path = tmp_path / "receipt.json"
    path.write_bytes(existing)
    with pytest.raises(ValueError, match="Existing receipt differs"):
        checks.save_receipt(path, checks.controls())
    assert path.read_bytes() == existing


def test_interrupted_publication_can_be_retried(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    report = checks.controls()
    real_link = checks.os.link

    def interrupted(*args, **kwargs):
        raise OSError("publication interrupted")

    monkeypatch.setattr(checks.os, "link", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        checks.save_receipt(path, report)
    assert not path.exists()
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(checks.os, "link", real_link)
    assert checks.save_receipt(path, report) == "written"


def test_check_failures_are_exceptions_not_disabled_assertions():
    with pytest.raises(ValueError, match="failed control"):
        checks.require(False, "failed control")
