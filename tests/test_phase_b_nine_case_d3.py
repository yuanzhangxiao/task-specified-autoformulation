"""D3 and HDTwinGen campaigns on the nine staged Phase B cases.

tests/test_d3_upstream.py and tests/test_hdtwingen.py cover the loop. What is
under test here is the campaign around it: plans name only the nine compared
cells with their recorded prompts, freeze against the verified staged inputs,
run the method the plan names against the endpoint it was frozen for, and
report every task.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autoformalism.baselines import d3_upstream
from autoformalism.baselines.d3_native import NativeD3Fit
from autoformalism.baselines.hdtwingen import HybridModelReply
from autoformalism.rebuttal import phase_b_d3
from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.rebuttal import phase_b_nine_case_d3 as campaign
from autoformalism.rebuttal.phase_c_vendored_campaign import (
    JETSTREAM2_HOSTED_BASE_URL,
    OUTAGE_PATIENCE_SECONDS,
)
from tests.test_phase_b_nine_case import ALIEN, build

MODEL = "served-model"
LOCAL = "http://127.0.0.1:8000"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _payload(config: Path, **overrides) -> dict:
    recorded = json.loads(config.read_text())["cells"][ALIEN]
    payload = {
        "schema_version": "phase-b-nine-case-d3-plan-1",
        "status": "frozen_before_calls",
        "development_only": True,
        "inputs_protocol": "phase-b-nine-case-inputs-1",
        "purpose": "Tests.",
        "method": "d3",
        "endpoint": "vm_local_vllm",
        "model": MODEL,
        "cells": [
            {
                "benchmark_id": ALIEN,
                "tier": "easy",
                "public_prompt_sha256": recorded["files"]["proposer_prompt.txt"],
            }
        ],
        "repetitions": [0, 1],
        "generations": 2,
        "patience": 2,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    payload.update(overrides)
    return payload


def _model(rhs: str = "-k * v01 + u01", networks=()) -> dict:
    reply = {
        "model_description": f"model: {rhs}",
        "model": {
            "candidate_id": "m",
            "states": [
                {"name": name, "kind": "observed", "observed_channel": name,
                 "rhs": rhs if name == "v01" else "0"}
                for name in ("v01", "v02", "v03")
            ],
            "parameters": [{"name": "k"}],
        },
        "parameter_initial_values": [{"name": "k", "value": 0.4}],
    }
    if networks:
        reply["networks"] = list(networks)
    return reply


class _Chat:
    """The loop's two calls, scripted."""

    def __init__(self, models, reply_type=d3_upstream.D3ModelReply) -> None:
        self.models = iter(models)
        self.reply_type = reply_type
        self.calls: list[tuple[str, int]] = []

    def reflect(self, messages, *, generation):
        self.calls.append(("reflect", generation))
        return "reflect"

    def write_model(self, messages, *, generation):
        self.calls.append(("model", generation))
        return self.reply_type.model_validate(next(self.models))


@pytest.fixture
def freeze(tmp_path, monkeypatch):
    """Freeze plans on staged synthetic inputs, with the torch fit stood in for."""
    source, config = build(tmp_path)
    monkeypatch.setattr(nine, "CONFIG", config)
    inputs = tmp_path / "staged"
    nine.stage([source], inputs)
    monkeypatch.setattr(
        campaign, "environment", lambda endpoint: {"code": "fixed", "kind": endpoint}
    )
    fitted: list[dict] = []

    def fit(*args, **kwargs):
        fitted.append(kwargs)
        weights = {"r": "w"} if "networks" in kwargs else None
        return NativeD3Fit({"k": 0.1}, 0.2, 0.1, 10, {"v01": 1.0}, weights)

    monkeypatch.setattr(d3_upstream, "fit_native_d3", fit)
    monkeypatch.setattr(
        d3_upstream, "raw_state_losses", lambda *a, **k: (0.05, {"v01": 0.05})
    )
    evaluated: list[dict] = []

    def evaluate(*args, **kwargs):
        evaluated.append(kwargs)
        return {
            "saved_one_step_matches": True,
            "native_one_step": {"normalized_mse": 0.1},
            "phase_b_rollout": {"status": "complete", "normalized_mse": 0.3},
        }

    monkeypatch.setattr(phase_b_d3, "evaluate_validation", evaluate)

    def frozen(**overrides) -> tuple[Path, dict]:
        plan = tmp_path / "plan.json"
        plan.write_text(json.dumps(_payload(config, **overrides)))
        root = tmp_path / f"campaign-{overrides.get('method', 'd3')}"
        return root, campaign.prepare(plan, inputs, root)

    frozen.fitted = fitted
    frozen.evaluated = evaluated
    frozen.config = config
    return frozen


def _run(root: Path, index: int = 0, **options) -> dict:
    options = {
        "endpoint": "vm_local_vllm",
        "base_url": LOCAL,
        "patience_seconds": 15 * 60.0,
        **options,
    }
    return campaign.run(root, index, **options)


def test_a_plan_freezes_rows_against_the_staged_inputs(freeze):
    _, sealed = freeze()
    assert [(r["benchmark_id"], r["repetition"]) for r in sealed["rows"]] == [
        (ALIEN, 0),
        (ALIEN, 1),
    ]
    assert sealed["method"] == "d3"
    assert sealed["maximum_logical_calls"] == 2 * (2 * 2 - 1)
    assert sealed["test_data_opened"] is False
    assert sealed["upstream_revision"] == d3_upstream.D3_UPSTREAM_REVISION
    assert any("white-box mode" in item for item in sealed["departures"])
    assert sealed["rows"][0]["public_identity"]["inputs_receipt_sha256"] == (
        sealed["inputs_receipt_sha256"]
    )


def test_an_hdtwingen_plan_records_its_own_release_and_departures(freeze):
    _, sealed = freeze(method="hdtwingen")
    assert sealed["upstream_revision"] == "a7cd335"
    assert sealed["loop"] == "hdtwingen_nsdt"
    assert any("multilayer perceptron" in item for item in sealed["departures"])


@pytest.mark.parametrize(
    "change",
    [
        {"tier": "hard"},
        {"benchmark_id": "phase_b_dalla_man_t3_canonical_named_easy"},
        {"public_prompt_sha256": "0" * 64},
    ],
)
def test_a_cell_must_be_one_of_the_nine_with_its_tier_and_prompt(
    freeze, tmp_path, change
):
    payload = _payload(freeze.config)
    payload["cells"][0].update(change)
    with pytest.raises(ValueError):
        campaign.NineCaseD3Plan.model_validate(payload)


def test_the_hosted_endpoint_must_name_its_one_model(freeze):
    payload = _payload(freeze.config, endpoint="jetstream2_hosted")
    with pytest.raises(ValueError, match="serves only"):
        campaign.NineCaseD3Plan.model_validate(payload)


def test_d3_runs_its_white_box_loop_and_resumes_without_a_call(freeze):
    root, _ = freeze()
    chat = _Chat([_model(), _model("-k * v01")])
    result = _run(root, chat=chat)
    assert result["status"] == "complete"
    assert result["selection_metric"] == d3_upstream.SELECTION
    assert chat.calls == [("model", 0), ("reflect", 1), ("model", 1)]
    assert all("networks" not in item for item in freeze.fitted)
    assert freeze.evaluated[-1]["networks"] == ()
    again = _Chat([])
    assert _run(root, chat=again) == result
    assert again.calls == []


def test_hdtwingen_fits_declared_networks_and_seals_their_weights(freeze):
    root, _ = freeze(method="hdtwingen")
    network = {
        "name": "r",
        "inputs": ["v01", "u01"],
        "hidden_layers": 1,
        "hidden_width": 4,
        "activation": "relu",
    }
    chat = _Chat(
        [_model("-k * v01 + r", networks=[network]), _model("-k * v01")],
        reply_type=HybridModelReply,
    )
    result = _run(root, chat=chat)
    assert result["status"] == "complete"
    assert "networks" in freeze.fitted[0]
    assert [spec.name for spec in freeze.evaluated[-1]["networks"]] == ["r"]
    assert freeze.evaluated[-1]["network_weights"] == {"r": "w"}


def test_a_task_runs_only_against_its_endpoint_and_code(freeze, monkeypatch):
    root, _ = freeze()
    with pytest.raises(ValueError, match="endpoint kind differs"):
        _run(root, endpoint="jetstream2_hosted", base_url=JETSTREAM2_HOSTED_BASE_URL)
    monkeypatch.setattr(campaign, "environment", lambda endpoint: {"code": "new"})
    with pytest.raises(ValueError, match="changed since this plan was frozen"):
        _run(root, chat=_Chat([]))


def test_changed_staged_inputs_are_refused_before_any_call(freeze):
    root, sealed = freeze()
    path = Path(sealed["inputs"]) / "phase_b_v1" / ALIEN / "validation.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs from the recorded file"):
        _run(root, chat=_Chat([]))


def test_the_served_model_is_checked_before_the_first_call(freeze, monkeypatch):
    root, _ = freeze()
    monkeypatch.setattr(campaign, "served_model_ids", lambda base_url: ("other",))
    with pytest.raises(ValueError, match="not among the served models"):
        _run(root)


def test_the_report_keeps_pending_tasks_and_states_the_limitation(freeze):
    root, _ = freeze(method="hdtwingen")
    _run(root, chat=_Chat([_model(), _model("-k * v01")],
                          reply_type=HybridModelReply))
    summary = campaign.report(root)
    assert summary["counts"] == {"complete": 1, "pending": 1}
    assert summary["status"] == "pending"
    assert summary["limitation"].startswith("HDTwinGen on the nine Phase B cases")


def test_the_cli_gives_a_task_its_endpoints_patience_and_cache_rule(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import phase_b_nine_case_d3 as cli

    seen = {}

    def run(root, index, **options):
        seen.update(options)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run", run)
    monkeypatch.setenv("AF_ENDPOINT_KIND", "jetstream2_hosted")
    monkeypatch.setattr(sys, "argv", ["cli.py", "run", "--root", "r", "--index", "0"])
    cli.main()
    assert seen == {
        "endpoint": "jetstream2_hosted",
        "base_url": JETSTREAM2_HOSTED_BASE_URL,
        "patience_seconds": OUTAGE_PATIENCE_SECONDS["jetstream2_hosted"],
        "bypass_cache": True,
    }


@pytest.mark.parametrize(
    ("name", "method", "cells", "repetitions", "generations"),
    [
        ("phase_b_nine_case_d3_smoke_v1.json", "d3", 1, (0,), 2),
        ("phase_b_nine_case_hdtwingen_smoke_v1.json", "hdtwingen", 1, (0,), 3),
        ("phase_b_nine_case_d3_hosted_120b_v1.json", "d3", 9, (0, 1, 2), 20),
        ("phase_b_nine_case_hdtwingen_hosted_120b_v1.json", "hdtwingen", 9,
         (0, 1, 2), 20),
    ],
)
def test_the_committed_plans_keep_upstreams_settings(
    name, method, cells, repetitions, generations
):
    plan = campaign.load_plan(CONFIGS / name)
    assert plan.method == method
    assert len(plan.cells) == cells
    assert plan.repetitions == repetitions
    assert plan.generations == plan.patience == generations
    assert plan.keep_top_samples == 16
    assert (plan.temperature, plan.top_p) == (0.7, 0.95)
    assert plan.reasoning_effort is None
    assert plan.endpoint == "jetstream2_hosted"
    assert plan.model == "gpt-oss-120b"


def test_a_smoke_clears_only_when_its_summary_is_complete_without_failures(freeze):
    root, _ = freeze()
    summary = campaign.report(root)
    assert nine.smoke_problems({"d3": summary}) == ["d3: report status pending"]
    for index in (0, 1):
        _run(root, index, chat=_Chat([_model(), _model("-k * v01")]))
    summary = campaign.report(root)
    assert summary == json.loads((root / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert nine.smoke_problems({"d3": summary}) == []
    # The CLI's report.json leaves the rows out; it cannot clear a smoke.
    without_rows = {key: value for key, value in summary.items() if key != "rows"}
    assert nine.smoke_problems({"d3": without_rows}) == [
        "d3: the report has no task rows"
    ]


def test_the_smoke_check_script_exits_by_the_summaries(freeze, tmp_path):
    import subprocess

    root, _ = freeze()
    for index in (0, 1):
        _run(root, index, chat=_Chat([_model(), _model("-k * v01")]))
    campaign.report(root)
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_phase_b_smokes.py"
    environment = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PATH": "/usr/bin:/bin",
    }
    passed = subprocess.run(
        [sys.executable, str(script), str(root / "summary.json")],
        capture_output=True, text=True, env=environment, check=False,
    )
    assert passed.returncode == 0, passed.stderr
    assert passed.stdout.strip().endswith("every smoke passed")
    stale = tmp_path / "stale" / "report.json"
    stale.parent.mkdir()
    summary = json.loads((root / "summary.json").read_text())
    stale.write_text(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    stopped = subprocess.run(
        [sys.executable, str(script), str(stale)],
        capture_output=True, text=True, env=environment, check=False,
    )
    assert stopped.returncode == 1
    assert "no campaign was started" in stopped.stdout
