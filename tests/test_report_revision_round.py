"""The report preserves missing picks, repeated slots and frozen model identities."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import report_revision_round as report


def test_readable_report_binds_frozen_receipts_and_escapes_edits(tmp_path):
    p = sealed_write(tmp_path / "plan.json", {"seeds": [100, 101, 102]})
    cases, choices, scores = {}, {}, {}
    receipt = {"status": "responded", "usage": {"total_tokens": 60}, "seconds": 0.5}
    for case in report.CASES:
        pool = ["c", "c", None]
        cases[case] = {
            "parent": "p",
            "slots": {"algorithm": ["c"], "20b_all_nine": pool},
            "fits": {
                "p": {"eligible": True, "train_nmse": 0.2},
                "c": {"eligible": True, "train_nmse": 0.1},
            },
            "eligible": {"p": {"name": "parent"}, "c": {"name": "<b>child</b>"}},
            "screen": {"screened": 3},
        }
        choices[case] = {
            arm: {
                "algorithm": "c",
                "20b": {"100": "c", "101": "p", "102": None},
                "uniform": {"100": "p"},
                "distinct_eligible_edits": 1,
                "eligible_slots": 1 if arm == "algorithm" else 2,
            }
            for arm in cases[case]["slots"]
        }
        scores[case] = {"p": {"nmse": 0.2}, "c": {"nmse": 0.3}}
    training = sealed_write(
        tmp_path / "training.json",
        {
            "plan": p["artifact_sha256"],
            "cases": cases,
            "generation_cost": {"r": receipt},
        },
    )
    frozen = sealed_write(
        tmp_path / "frozen.json",
        {
            "training": training["artifact_sha256"],
            "choices": choices,
            "rank_cost": {"r": {**receipt, "permutation_valid": True}},
        },
    )
    sealed_write(
        tmp_path / "development.json",
        {"freeze": frozen["artifact_sha256"], "cases": scores},
    )
    html = report.render(tmp_path).read_text()
    assert html.count("<svg") == 3
    assert "&lt;b&gt;child&lt;/b&gt;" in html and "<b>child</b>" not in html
    assert "1 of 1 rank replies gave a full permutation" in html
    assert "higher development NMSE" in html and "M1 (0.3)" in html
    # A separately sealed, mismatched decision chain is still rejected.
    (tmp_path / "development.json").unlink()
    sealed_write(
        tmp_path / "development.json", {"freeze": "unrelated", "cases": scores}
    )
    with pytest.raises(ValueError, match="receipts disagree"):
        report.render(tmp_path)
