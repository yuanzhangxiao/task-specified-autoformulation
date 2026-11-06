"""Portable admission, matched budgets and nonrenewing revision-fit execution."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import insulin_revision_fitting as c
from tests.test_prefix_portfolio_root_profile import changed


def policy():
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=2,
        maximum_calls=5,
        qualification_seconds=3,
        verification_seconds=3,
        persistence_headroom_seconds=0,
    )


def inputs():
    source = changed()
    parent = source.problem.model_dump(mode="json")
    revised = deepcopy(parent)
    revised["base_candidate"]["parameters"].append(
        {"name": c.DELAY, "scope": "global", "role": "coefficient"}
    )
    revised["base_candidate"]["state_equations"][0]["rhs"] += f" + {c.DELAY} * z"
    revised["parameter_guesses"][c.DELAY] = 1.0
    revised = c.PrefixFitProblem.model_validate(revised).model_dump(mode="json")
    training = source.training.model_dump(mode="json")
    warm = source.start | {c.DELAY: 0.0}
    endpoint = {
        "parameters": source.start,
        "training_nmse": 1.0,
        "original_equations_verified": True,
        "status": "complete",
    }
    model_hash, train_hash = (
        c.public.content_sha256(revised),
        c.public.content_sha256(training),
    )
    docs = {
        "problem": revised,
        "parent_problem": parent,
        "training": training,
        "numerical": c.kernel.NumericalPolicy().model_dump(mode="json"),
        "warm_start_parameters": warm,
        "baseline_endpoint": endpoint,
        "fitting_contract": {
            "prefix_policy": {"duration": 1.0},
            "training_sha256": train_hash,
        },
        "admission": {"refit_handoff": {"problem_sha256": model_hash}},
        "nested_replay": {
            "problem_sha256": model_hash,
            "training_sha256": train_hash,
            "warm_start_sha256": c.public.content_sha256(warm),
            "old_endpoint_sha256": c.public.content_sha256(endpoint),
        },
        "proposal": {},
        "provenance": {"historical_fit": {"fixture": True}},
    }
    return {
        "protocol": c.INPUT_PROTOCOL,
        "manifest": {},
        "documents": docs,
        "test_data_opened": False,
        "validation_exported": False,
    }


class FakeFit:
    """Exercise durable orchestration; the numerical kernel has separate tests."""

    def __init__(self, data, *, loss=0.5, interrupted=False, failed=False):
        self.data, self.loss = data, loss
        self.interrupted, self.failed = interrupted, failed

    def setup_audit(self):
        return {"passed": True, "fixture": True}

    def verify_start(self, expected, seconds):
        return c._source_endpoint(self.data) | {"passed": True}

    def qualify(self, seconds, save):
        value = {
            "passed": not self.failed,
            "parameters": self.data["documents"]["warm_start_parameters"],
            "training_nmse": 1.0,
        }
        save(value)
        return value

    def search(self, settings, save):
        result = {
            "search_calls_completed": 4,
            "search_calls_started": 4,
            "stop_reason": "optimizer_stopped",
            "optimizer_success": True,
            "budget_exhausted": False,
            "optimizer_terminal": {"feasible_evaluation_verified": True},
            "best": {
                "parameters": self.data["documents"]["warm_start_parameters"],
                "training_nmse": self.loss,
            },
        }
        save(result)
        if self.interrupted:
            raise KeyboardInterrupt()
        return result

    def verify(self, point, loss, seconds, save):
        value = {
            "original_equations_verified": True,
            "parameters": point,
            "training_nmse": loss,
            "status": "complete",
        }
        save(value)
        return value

    def metrics(self, endpoint, seconds):
        return {
            "training_sha256": c.public.content_sha256(
                self.data["documents"]["training"]
            ),
            "training_nmse": endpoint["training_nmse"],
            "problem_sha256": c.public.content_sha256(
                self.data["documents"]["problem"]
            ),
            "rows": [],
        }


@pytest.fixture
def ready(tmp_path, monkeypatch):
    data = inputs()
    monkeypatch.setattr(c.public, "_source_identity", lambda: "frozen-source")
    c.prepare_inputs(tmp_path, data, policy())

    def setup(self, save):
        self.fit = FakeFit(self.data)
        result = self.fit.setup_audit()
        save(result)
        return result

    monkeypatch.setattr(c._Run, "setup", setup)
    monkeypatch.setattr(
        c.metrics,
        "compare",
        lambda a, b: {"improvement": a["training_nmse"] - b["training_nmse"]},
    )
    monkeypatch.setattr(c.metrics, "render_comparison", lambda *a: "fixture comparison")
    return tmp_path


def test_primary_and_robustness_tasks_have_separate_roles_and_fixed_signed_starts(
    ready,
):
    plan, _ = c.verify(ready)
    assert [t["role"] for t in plan["tasks"]] == [
        "primary_matched_budget",
        "additional_robustness",
        "additional_robustness",
    ]
    assert [t["delay_start"] for t in plan["tasks"]] == [
        0,
        -c.START_MULTIPLIER * c.FEEDBACK_UNIT,
        c.START_MULTIPLIER * c.FEEDBACK_UNIT,
    ]
    assert c.reservation_ceiling(c.Policy()) < 4 * 3600
    assert plan["matched_comparison"] == "primary_nested_zero_only"


def test_complete_fit_retains_verified_improvement_and_read_only_resume(
    ready, monkeypatch
):
    result = c.run_task(ready, 0)
    assert result["training_reconstruction_nmse"] == 0.5
    assert result["retained_origin"] == "endpoint_verification"
    assert result["search_status"] == "qualified_optimizer_stop"
    assert result["qualified_search_stop"]
    assert not result["optimization_converged"]
    assert result["comparison"]["comparison"]["improvement"] == 0.5
    before = {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("unexpected rebuild"))
    assert c.run_task(ready, 0) == result
    summary = c.report(ready)
    assert summary == c.report(ready)
    assert summary["recorded"] == 1 and summary["expected"] == 3
    assert [r["status"] for r in summary["rows"]] == ["complete", "missing", "missing"]
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}


@pytest.mark.parametrize("loss", [1.0, 1.2, 1.0 - 1e-12])
def test_baseline_survives_worse_equal_or_unresolved_trial(ready, monkeypatch, loss):
    real = FakeFit.search

    def search(self, p, save):
        self.loss = loss
        return real(self, p, save)

    monkeypatch.setattr(FakeFit, "search", search)
    result = c.run_task(ready, 0)
    assert result["training_reconstruction_nmse"] == 1.0
    assert result["retained_origin"] == "source_nested_baseline"


def test_failed_qualification_preserves_baseline_and_records_reason(ready, monkeypatch):
    monkeypatch.setattr(
        FakeFit, "qualify", lambda *a: {"passed": False, "reason": "bad design"}
    )
    result = c.run_task(ready, 0)
    assert result["search_status"] == "profile_unqualified"
    assert "search" not in result["operation_statuses"]
    assert result["training_reconstruction_nmse"] == 1.0
    assert result["comparison"] is not None


def test_started_setup_never_retried_and_charged(ready, monkeypatch):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 0)
    seal(
        run.folder / "operations/setup/started.json",
        {
            "campaign": run.identity,
            "stage": "setup",
            "allowance_seconds": run.allowances["setup"],
        },
    )
    monkeypatch.setattr(
        c._Run, "setup", lambda *a: pytest.fail("renewed started operation")
    )
    result = c.run_task(ready, 0)
    assert result["search_status"] == "setup_unavailable"
    assert result["training_reconstruction_nmse"] == 1.0
    assert result["cost"]["budget_charge_seconds"] == 180
    assert not result["cost"]["accounting_complete"]


def test_interrupted_search_progress_replayed_but_search_not_renewed(
    ready, monkeypatch
):
    original = FakeFit.search

    def interrupted(self, p, save):
        self.interrupted = True
        return original(self, p, save)

    monkeypatch.setattr(FakeFit, "search", interrupted)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 0)
    monkeypatch.setattr(FakeFit, "search", lambda *a: pytest.fail("renewed search"))
    result = c.run_task(ready, 0)
    assert result["search_status"] == "search_interrupted_or_unavailable"
    assert result["training_reconstruction_nmse"] == 0.5
    assert result["retained_origin"] == "endpoint_verification"
    assert not result["qualified_search_stop"]


def test_failed_trial_verification_cannot_displace_baseline(ready, monkeypatch):
    original = FakeFit.verify

    def verify(self, p, loss, seconds, save):
        value = original(self, p, loss, seconds, save)
        return value | {"original_equations_verified": loss == 1.0}

    monkeypatch.setattr(FakeFit, "verify", verify)
    result = c.run_task(ready, 0)
    assert result["training_reconstruction_nmse"] == 1.0


@pytest.mark.parametrize(
    "kind", ["extra", "test", "validation", "hash", "warm", "graph", "nonfinite"]
)
def test_intake_boundaries_fail_before_plan_publication(tmp_path, kind):
    data = inputs()
    d = data["documents"]
    if kind == "extra":
        data["private_parameters"] = {}
    elif kind == "test":
        d["training"]["name"] = "test"
    elif kind == "validation":
        data["validation_exported"] = True
    elif kind == "hash":
        d["admission"]["refit_handoff"]["problem_sha256"] = "0" * 64
    elif kind == "warm":
        d["warm_start_parameters"][c.DELAY] = 1.0
    elif kind == "graph":
        d["problem"]["graph"]["rules"]["x"]["initial"]["value"] = 2
    else:
        d["baseline_endpoint"]["training_nmse"] = float("inf")
    with pytest.raises(ValueError):
        c.prepare_inputs(tmp_path, data, policy())
    assert not (tmp_path / "plan.json").exists()


def test_production_projection_and_declared_coordinate_roster_are_pinned(tmp_path):
    with pytest.raises(ValueError, match="source projection"):
        c.prepare_inputs(tmp_path, inputs(), c.Policy())


def test_changed_policy_or_runtime_cannot_resume(ready, monkeypatch):
    before = read_seal(ready / "plan.json")
    monkeypatch.setattr(c.public, "_source_identity", lambda: "changed-source")
    with pytest.raises(ValueError, match="identity"):
        c.verify(ready)
    assert c.verify(ready, runtime=False)[0] == before
    plan = deepcopy(before)
    plan["matched_comparison"] = "best_of_three"
    (ready / "plan.json").unlink()
    seal(ready / "plan.json", plan)
    with pytest.raises(ValueError, match="identity"):
        c.verify(ready, runtime=False)


def test_foreign_metric_rows_and_false_score_are_rejected(ready):
    _, data = c.verify(ready)
    endpoint = c._source_endpoint(data)
    baseline = FakeFit(data).metrics(endpoint, 1)
    with pytest.raises(ValueError, match="different training"):
        c.compare_refit_replay(
            data, baseline, baseline | {"training_sha256": "foreign"}, endpoint
        )
    with pytest.raises(ValueError, match="replay differ"):
        c.compare_refit_replay(
            data, baseline, baseline | {"training_nmse": 0.1}, endpoint
        )


def test_receipt_or_checkpoint_change_is_rejected(ready):
    c.run_task(ready, 0)
    path = ready / "results/nested-zero/operations/search/progress.json"
    value = c.public._read(path)
    value["best"]["training_nmse"] = 0.01
    c.public._write(path, value)
    with pytest.raises(ValueError, match="checkpoint differs"):
        c.report(ready)


def test_invalid_worker_index_and_missing_report_do_not_fit(ready, monkeypatch):
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("fitted during report"))
    assert c.report(ready)["recorded"] == 0
    for index in (-1, 3, True):
        with pytest.raises(ValueError, match="index"):
            c.run_task(ready, index)


def test_source_manifest_unknown_member_or_file_hash_fails(tmp_path, monkeypatch):
    import json
    from hashlib import sha256

    source = tmp_path / "source"
    source.mkdir()
    data = inputs()
    files = {}
    for key, value in data["documents"].items():
        raw = json.dumps(value).encode()
        (source / (key + ".json")).write_bytes(raw)
        files[key + ".json"] = sha256(raw).hexdigest()
    (source / "HANDOFF.md").write_text("Fixture")
    files["HANDOFF.md"] = sha256(b"Fixture").hexdigest()
    manifest = {
        "protocol": "orion-insulin-revision-inputs-1",
        "files": files,
        "test_data_opened": False,
        "new_model_fitted": False,
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(c, "SOURCE_MANIFEST", c.public.content_sha256(manifest))
    assert c.load_intake(source)["documents"] == data["documents"]
    (source / "problem.json").write_text("{}")
    with pytest.raises(ValueError, match="file digest"):
        c.load_intake(source)
    manifest["files"]["../private.json"] = "0" * 64
    (source / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(c, "SOURCE_MANIFEST", c.public.content_sha256(manifest))
    with pytest.raises(ValueError, match="manifest differs"):
        c.load_intake(source)
