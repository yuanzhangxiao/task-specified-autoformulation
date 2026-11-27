"""Check sealed decision-report comparisons without loading benchmark data."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts.compare_evidence_decisions import design
from scripts.report_evidence_decisions import render


def fixture(root, *, wrong_result=False):
    cfg = design()
    plan = sealed_write(
        root / "plan.json",
        {
            "design": cfg,
            "basin_screens": {
                "0.0001": {
                    f: {"family_rejections": 0, "eligible_fits": 1}
                    for f in ("simple", "extended")
                }
            },
        },
    )
    decisions, rows = [], []
    for benchmark in ("basin", "t2"):
        for mode, transfer, budget in [cfg["primary"], ["precision_10mm", 0.0, 9]]:
            panel = benchmark + "/fold0"
            key = f"{panel}/{mode}"
            status = mode == "forecast"
            trace = [
                {
                    "spent": 0,
                    "action": "fit_simple" if status else "check_simple",
                    "stop_risk": 1.0,
                    "action_values": {"fit_simple": 0.5, "check_simple": 0.6},
                    "observation": "2" if status else "reject",
                },
                {
                    "spent": 4 if status else 1,
                    "action": "stop",
                    "stop_risk": 0.5,
                    "action_values": {},
                },
            ]
            receipt = sealed_write(
                root / "decisions" / f"{key}.json",
                {
                    "panel": panel,
                    "plan": plan["artifact_sha256"],
                    "benchmark": benchmark,
                    "mode": mode,
                    "transfer": transfer,
                    "budget": budget,
                    "decisions": {
                        "adaptive": {
                            "trace": trace,
                            "model": "simple" if status else None,
                            "checks": int(not status),
                        }
                    },
                },
            )
            decisions.append((key, receipt["artifact_sha256"]))
            for pol in cfg["policy"]:
                rows.append(
                    {
                        "policy": pol,
                        "benchmark": benchmark,
                        "panel": panel,
                        "mode": mode,
                        "transfer": transfer,
                        "budget": budget,
                        "validation_nmse": 0.12 if status else None,
                        "terminal_loss": 0.12 if status else 1.0,
                        "loss_with_cost": 0.24 if status else 1.03,
                        "fits": int(status),
                        "checks": int(not status),
                        "cost_units": 4 if status else 1,
                        "optimizer_evaluations_replayed": 10 if status else 0,
                    }
                )
    choices = sealed_write(
        root / "choices.json", {"plan": plan["artifact_sha256"], "receipts": decisions}
    )
    sealed_write(
        root / "results.json",
        {
            "choices": "other" if wrong_result else choices["artifact_sha256"],
            "rows": rows,
        },
    )


def test_report_uses_only_sealed_receipts_and_shows_abstention(tmp_path):
    fixture(tmp_path)
    text = render(tmp_path).read_text()
    assert "0/1" in text and "1/1" in text
    assert all(
        s in text for s in ("T2 disposal", "Coupled basin", "10 mm", "Abstained")
    )
    assert text.count("Decision receipt</a>") == 3
    assert "decisions/basin/fold0/precision_10mm.json" in text


def test_result_belongs_to_frozen_policy(tmp_path):
    fixture(tmp_path, wrong_result=True)
    with pytest.raises(ValueError, match="Unmatched"):
        render(tmp_path)
