"""M13 allocation, independent evaluation boundaries and failed-worker recovery."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import portfolio_campaign as campaign
from autoformalism.fitting import portfolio_fit as fitting
from autoformalism.fitting import portfolio_starts as starts
from autoformalism.fitting import public_fitting as public
from tests.test_generic_recovery import saved as source_fixture


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return source_fixture.__wrapped__(tmp_path, monkeypatch)


def policy():
    return campaign.PortfolioPolicy(
        seconds=90,
        certificate_seconds=15,
        native_seconds=10,
        point_seconds=3,
        probe_seconds=10,
        continuation_seconds=15,
        probe_calls=10,
        continuation_calls=15,
        maximum_rollout_calls=80,
        minimum_intervals=1,
        medium_target=400,
        targets=(12, None),
    )


def base(saved):
    return common.bases(read_seal(saved))["linear_s0"]


def test_starts_are_repeatable_bounded_and_no_reference_dependent(saved):
    b = base(saved)
    points = starts.generate(b, 0.7)
    assert len(points) == 3 and points[0]["parameters"] == b["start"]
    assert all(starts.valid(p["parameters"], starts.domain(b)) for p in points)
    changed = deepcopy(read_seal(saved))
    changed["case"]["reference_parameters"] = {"a": 999}
    changed["case"]["validation"] = {"poison": True}
    assert points == starts.generate(common.bases(changed)["linear_s0"], 0.7)
    assert points == starts.generate(b, 0.7)
    assert points[1]["parameters"] != points[2]["parameters"]
    b["training"]["name"] = "val"
    with pytest.raises(ValueError):
        starts.generate(b, 0.7)


@pytest.mark.parametrize(
    "changes",
    [
        {"seconds": 100, "probe_seconds": 120},
        {"maximum_rollout_calls": 20},
        {"perturbation_spread": 0},
    ],
)
def test_invalid_policy_cannot_starve_trials(changes):
    values = policy().model_dump() | changes
    with pytest.raises(ValueError):
        campaign.PortfolioPolicy.model_validate(values)


def invoke_fake(calls, *, failures=(), timeout=False, uncertified=False):
    """Complete training evidence, or explicit derivative failure at a known trial."""

    def invoke(mode, payload, folder, seconds):
        calls.append((mode, payload))
        folder.mkdir(parents=True, exist_ok=True)
        value = {"payload_sha256": public.content_sha256(payload)}
        if mode == "recovery_check":
            value.update(parameters=payload["parameters"], passed=not uncertified)
        else:
            assert mode == "recovery_rollout"
            (folder / "refinement").mkdir(exist_ok=True)
            point = payload["points"][0]
            initial = not any(
                p["points"][0]["source"] == point["source"]
                for m, p in calls[:-1]
                if m == mode
            )
            failed = point["source"] in failures
            score = 0.8 if point["source"] == "generic_start" else 0.5
            # Successful later refinement from the originally worse pair.
            score = score if initial or failed else 1e-10
            reason = (
                "point_allowance_exhausted"
                if failed
                else (
                    "block_allowance_exhausted"
                    if initial
                    else "training_accuracy_reached_pending_replay"
                )
            )
            value.update(
                parameters=point["parameters"],
                training_nmse=score,
                actual_residual_calls=2,
                stop_reason=reason,
            )
            public._write(folder / "refinement/stages.json", [{"stop_reason": reason}])
            public._write(
                folder / "refinement/best.json",
                {"parameters": point["parameters"], "training_nmse": score},
            )
        if not timeout:
            public._write(folder / "result.json", value)
        return {
            "status": "timeout" if timeout else "complete",
            "termination_confirmed": True,
            "elapsed_seconds": 1,
        }

    return invoke


def test_every_start_gets_trial_and_derivative_failure_is_local(
    saved, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        fitting.process,
        "invoke",
        invoke_fake(
            calls, failures=("generic_perturbation_0", "generic_perturbation_1")
        ),
    )
    result = fitting.fit(
        base(saved), "rollout_portfolio", policy().model_dump(), tmp_path
    )
    sources = [p["points"][0]["source"] for m, p in calls if m == "recovery_rollout"]
    assert sources[:3] == [
        "generic_start",
        "generic_perturbation_0",
        "generic_perturbation_1",
    ]
    assert sources[3] == "generic_start"
    assert result["stop_reason"] == "training_prediction_certified"
    assert result["selected"]["parameters"] == base(saved)["start"]
    assert all(
        c["retired"] == "point_allowance_exhausted" for c in result["portfolio"][1:]
    )
    assert result["rollout_calls_charged"] == 8


def test_failed_best_trial_does_not_erase_incumbent(saved, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        fitting.process,
        "invoke",
        invoke_fake(
            calls,
            failures=(
                "generic_start",
                "generic_perturbation_0",
                "generic_perturbation_1",
            ),
            uncertified=True,
        ),
    )
    r = fitting.fit(base(saved), "rollout_portfolio", policy().model_dump(), tmp_path)
    assert r["selected"]["training_nmse"] == 0.5
    assert all(c["retired"] for c in r["portfolio"])
    assert len([m for m, _ in calls if m == "recovery_rollout"]) == 3


def test_lower_loss_checkpoint_never_displaces_generic_trial(
    saved, tmp_path, monkeypatch
):
    b = base(saved)
    checkpoint = starts.generate(b, 0.3)[1]["parameters"]
    monkeypatch.setattr(
        fitting.Portfolio,
        "collocation",
        lambda self: {
            "parameters": checkpoint,
            "training_nmse": 0.01,
            "origin": "collocation",
        },
    )
    calls = []
    monkeypatch.setattr(fitting.process, "invoke", invoke_fake(calls))
    r = fitting.fit(b, "checkpoint_portfolio", policy().model_dump(), tmp_path)
    sources = [p["points"][0]["source"] for m, p in calls if m == "recovery_rollout"]
    assert sources[:3] == ["generic_start", "collocation", "generic_perturbation_1"]
    assert r["portfolio"][0]["parameters"] == b["start"]


def test_timeout_counts_unknown_calls_at_cap_and_keeps_checkpoint(
    saved, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(fitting.process, "invoke", invoke_fake(calls, timeout=True))
    r = fitting.fit(base(saved), "rollout_portfolio", policy().model_dump(), tmp_path)
    assert r["selected"] is not None and r["rollout_calls_charged"] <= 80
    assert all(x["unknown_calls_charged_at_cap"] for x in r["rollout_call_accounting"])
    assert r["stop_reason"] != "training_prediction_certified"


def test_unconfirmed_cleanup_stops_all_further_work(saved, tmp_path, monkeypatch):
    calls = []

    def invoke(*args):
        calls.append(args)
        return {"status": "cleanup_unconfirmed", "termination_confirmed": False}

    monkeypatch.setattr(fitting.process, "invoke", invoke)
    r = fitting.fit(base(saved), "rollout_portfolio", policy().model_dump(), tmp_path)
    assert len(calls) == 1 and r["stop_reason"] == "cleanup_unconfirmed"


def test_interruption_cannot_refresh_budget(saved, tmp_path, monkeypatch):
    b, p = base(saved), policy().model_dump()
    identity = public.content_sha256(
        {"base": b, "arm": "rollout_portfolio", "policy": p}
    )
    seal(tmp_path / "fit-started.json", {"identity": identity})
    monkeypatch.setattr(fitting.process, "invoke", lambda *a: pytest.fail("restarted"))
    r = fitting.fit(b, "rollout_portfolio", p, tmp_path)
    assert r["stop_reason"] == "interrupted_no_fit_restart"
    with pytest.raises(ValueError, match="identity"):
        fitting.fit(b, "rollout_portfolio", p | {"seconds": 91}, tmp_path)


def test_certificate_rejection_does_not_loop_on_same_pair(saved, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fitting.process, "invoke", invoke_fake(calls, uncertified=True))
    r = fitting.fit(base(saved), "rollout_portfolio", policy().model_dump(), tmp_path)
    assert r["stop_reason"] != "training_prediction_certified"
    assert all(c["retired"] == "training_stop_not_certified" for c in r["portfolio"])
    assert len([m for m, _ in calls if m == "recovery_rollout"]) == 6


def test_call_accounting_and_worker_identity_fail_closed(saved, tmp_path, monkeypatch):
    def invoke(mode, payload, folder, seconds):
        folder.mkdir(parents=True, exist_ok=True)
        public._write(folder / "result.json", {"payload_sha256": "different"})
        return {"termination_confirmed": True, "status": "complete"}

    monkeypatch.setattr(fitting.process, "invoke", invoke)
    with pytest.raises(ValueError, match="identity"):
        fitting.fit(base(saved), "rollout_portfolio", policy().model_dump(), tmp_path)


def test_campaign_freezes_selection_before_evaluation(saved, tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    campaign.prepare(root, saved, policy())
    with pytest.raises(ValueError, match="matching"):
        common.verify(root)
    monkeypatch.setattr(
        fitting,
        "fit",
        lambda *a: {
            "selected": {"parameters": base(saved)["start"]},
            "levels": [],
            "portfolio": [],
            "stop_reason": "portfolio_or_budget_stop",
        },
    )

    def evaluation(folder, *a):
        assert (folder.parent / "backend.json").exists()
        return {"status": "complete", "accuracy_passed": False}

    monkeypatch.setattr(common.replay, "_evaluation", evaluation)
    result = campaign.run_task(root, 0)
    assert campaign.run_task(root, 0) == result
    r = campaign.report(root)
    assert r["recorded"] == 1 and r["expected"] == 3
    assert r["protocol"] == campaign.PROTOCOL


def test_scheduler_and_inspector_use_new_protocol(saved, tmp_path, monkeypatch):
    from scripts import submit_phase_c_start_portfolio as submit

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        submit.scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submit.scheduler.subprocess, "run", sbatch)
    config = tmp_path / "policy.json"
    config.write_text(policy().model_dump_json())
    root = tmp_path / "submit"
    a = submit.submit(root, config, account="test", concurrency=2, inputs=saved)
    assert "--array=0-2%2" in calls[1] and "--job-name=portfolio-fit-run" in calls[1]
    assert submit.submit(root, config, account="test", concurrency=2, inputs=saved) == a
    assert len(calls) == 3


def test_shell_scripts_parse_and_reject_old_campaign():
    for stage in ("submit", "run", "inspect"):
        path = Path(f"scripts/hpc/{stage}_phase_c_start_portfolio_delta.sh")
        subprocess.run(["bash", "-n", str(path)], check=True)
    text = Path("scripts/hpc/inspect_phase_c_start_portfolio_delta.sh").read_text()
    assert campaign.PROTOCOL in text and "AF_CAMPAIGN" not in text
