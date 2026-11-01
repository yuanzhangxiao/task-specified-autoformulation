"""Source authentication, separate endpoint histories, and durable bounded work."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.research import construction_portfolio_search as campaign
from tests.test_construction_portfolio_continuation import continued  # noqa: F401
from tests.test_construction_portfolio_fitting import completed  # noqa: F401
from tests.test_construction_portfolio_recovery import recovered  # noqa: F401


def policy():
    return campaign.Policy(
        population="synthetic-smoke",
        prerequisite_seconds=10,
        coordinate_seconds=10,
        search_seconds=10,
        maximum_calls=30,
        verification_seconds=10,
        diagnostic_seconds=10,
        profile_seconds=1,
        profile_calls=4,
    )


@pytest.fixture
def prepared(continued, tmp_path):  # noqa: F811
    root = tmp_path / "search"
    before = {p: p.read_bytes() for p in continued[0].rglob("*.json")}
    campaign.prepare(root, continued[0], policy())
    assert before == {p: p.read_bytes() for p in continued[0].rglob("*.json")}
    return root


def install_profile_controls(monkeypatch):
    """Only substitute unsupported toy chart construction; use actual ODE replay."""
    monkeypatch.setattr(timescale, "make_oracle", campaign.polish.make_oracle)
    monkeypatch.setattr(
        timescale,
        "chart_equivalence",
        lambda original, oracle, seconds, save: campaign.previous._saved(
            {"passed": True, "toy_chart_control": True}, save
        ),
    )
    monkeypatch.setattr(timescale, "profile_grid", lambda oracle: list(range(10)))

    def search(oracle, spec, seconds, calls, save):
        return campaign.previous._saved(
            {
                "profile": spec,
                "best": {"parameters": oracle.start, "training_nmse": 0},
                "toy_journal_control": True,
            },
            save,
        )

    monkeypatch.setattr(timescale, "profile_search", search)


def test_complete_report_resume_and_profiles_never_select(prepared, monkeypatch):
    install_profile_controls(monkeypatch)
    results = [campaign.run_task(prepared, j) for j in range(2)]
    for result in results:
        assert result["search_status"] == "timescale_profiles_recorded"
        assert result["retained_origin"] == "source"
        assert result["retained"] == result["source_endpoint"]
        assert not result["profile_endpoints_used_for_selection"]
        assert len(result["profiles"]) == 10
        assert all(
            p["verification"]["original_equations_verified"] for p in result["profiles"]
        )
        assert result["retained_verification"]["original_equations_verified"]
        # A perfect toy incumbent can terminate before any accepted optimizer step.
        if result["accepted_verification"] is not None:
            assert result["accepted_verification"]["original_equations_verified"]
        assert not result["training_prefix_reestimated"]
    before = {p: p.read_bytes() for p in (prepared / "results").rglob("*.json")}
    monkeypatch.setattr(campaign._Run, "setup", lambda *a: pytest.fail("recompiled"))
    assert campaign.run_task(prepared, 0) == results[0]
    summary = campaign.report(prepared)
    assert summary == campaign.report(prepared)
    assert summary["recorded"] == 2
    assert campaign.inspection(prepared).is_file()
    assert before == {p: p.read_bytes() for p in (prepared / "results").rglob("*.json")}


def test_failed_chart_never_spends_profile_allowances(prepared, monkeypatch):
    monkeypatch.setattr(campaign._Run, "endpoints", lambda *args: None)
    monkeypatch.setattr(
        timescale,
        "make_oracle",
        lambda *args: (_ for _ in ()).throw(ValueError("unsupported chart")),
    )
    result = campaign.run_task(prepared, 0)
    assert result["search_status"] == "timescale_diagnostic_unavailable"
    assert result["operation_statuses"]["chart_qualification"] == "unavailable"
    assert not any(k.startswith("profile_") for k in result["operation_statuses"])
    assert result["retained"] == result["source_endpoint"]


def test_profile_interruption_is_charged_without_repeat(prepared, monkeypatch):
    install_profile_controls(monkeypatch)
    monkeypatch.setattr(campaign._Run, "endpoints", lambda *args: None)
    original = timescale.profile_search
    calls = []

    def interrupted(oracle, spec, seconds, maximum_calls, save):
        calls.append(spec)
        value = original(oracle, spec, seconds, maximum_calls, save)
        if spec == 0:
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr(timescale, "profile_search", interrupted)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(prepared, 0)
    result = campaign.run_task(prepared, 0)
    assert calls == list(range(10))
    assert result["operation_statuses"]["profile_00_search"] == "interrupted"
    assert result == campaign.report(prepared)["rows"][0]


def test_source_roster_and_scientific_tampering_rejected(continued, tmp_path):  # noqa: F811
    source = continued[0]
    with pytest.raises(ValueError, match="separate"):
        campaign.prepare(source / "nested", source, policy())
    data = campaign._source_inputs(source, policy())
    changed = deepcopy(data)
    changed["records"][0]["source_terminal"]["training_reconstruction_nmse"] += 1e-12
    with pytest.raises(ValueError, match="terminal or authentication"):
        campaign.prepare_inputs(tmp_path / "tampered", changed, policy())
    changed = deepcopy(data)
    changed["records"].reverse()
    with pytest.raises(ValueError, match="order"):
        campaign.prepare_inputs(tmp_path / "reversed", changed, policy())
    changed = deepcopy(data)
    changed["records"][0]["operations"]["search"]["identity"]["allowance_seconds"] += 1
    with pytest.raises(ValueError, match="operation identity"):
        campaign.prepare_inputs(tmp_path / "allowance", changed, policy())
    with pytest.raises(ValueError, match="reviewed"):
        campaign.prepare_inputs(tmp_path / "production", data, campaign.Policy())


def test_runtime_and_terminal_scientific_checks(prepared, monkeypatch):
    install_profile_controls(monkeypatch)
    monkeypatch.setattr(campaign._Run, "endpoints", lambda *args: None)
    campaign.run_task(prepared, 0)
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.run_task(prepared, 0)
    assert campaign.report(prepared)["recorded"] == 1
    path = prepared / "results/candidate-00/result.json"
    value = campaign.read_seal(path)
    value["training_reconstruction_nmse"] += 1e-12
    path.unlink()
    campaign.seal(path, value)
    with pytest.raises(ValueError, match="outside derived timing"):
        campaign.report(prepared)


def test_retention_needs_qualification_and_independent_replay():
    def endpoint(loss):
        return {
            "status": "complete",
            "training_nmse": loss,
            "original_equations_verified": True,
        }

    def op(value):
        return {"status": "complete", "value": value}

    record = {"source_terminal": {"retained": endpoint(5)}}
    operations = {
        "qualification": op({"search_qualified": True}),
        "search": op({"best": {"parameters": {"x": 1}}}),
        "start_verification": op(endpoint(6)),
        "endpoint_verification": op(endpoint(4)),
        "profile_00_verification": op(endpoint(0)),
    }
    assert campaign._retained(record, operations) == ("bounded_search", endpoint(4))
    for key in (
        "qualification",
        "search",
        "start_verification",
        "endpoint_verification",
    ):
        assert campaign._retained(
            record, {k: v for k, v in operations.items() if k != key}
        ) == ("source", endpoint(5))
    assert campaign._retained(
        {"source_terminal": {"retained": None}}, {"start_verification": op(endpoint(6))}
    ) == ("verified_start", endpoint(6))


def test_endpoint_history_is_never_relabelled(prepared, monkeypatch):
    plan, data = campaign.verify(prepared)
    run = campaign._Run(prepared, plan, data, 0)
    run.setup(lambda value: None)
    history = campaign.first._value(run.record["operations"]["search"])
    accepted = dict(history.get("last_accepted_parameters") or run.oracle.start)
    key = next(iter(accepted))
    accepted[key] += 1e-5
    history = history | {"last_accepted_parameters": accepted}
    seen = []

    def diagnosis(original, oracle, parameters, verification, received, seconds, save):
        seen.append((dict(parameters), deepcopy(received)))
        return campaign.previous._saved({"assessment": "toy_control"}, save)

    monkeypatch.setattr(campaign.numerical, "diagnose", diagnosis)
    run.endpoints(history)
    assert len(seen) == 2
    assert seen[0][0] != seen[1][0]
    assert seen[1][0] == accepted
    assert seen[0][1] == seen[1][1] == history
    # The backend consumes history only after exact endpoint equality.


def test_targeted_precision_shares_prerequisites_and_authenticates_probes(
    prepared, monkeypatch
):
    plan, data = campaign.verify(prepared)
    run = campaign._Run(prepared, plan, data, 0)
    run.task = run.task | {"kind": "targeted_precision"}
    run.allowances = campaign._allowances(run.task, run.policy)
    run.oracle = SimpleNamespace()
    calls = []

    def replay(oracle, method, seconds, save):
        calls.append(method)
        return campaign.previous._saved({"passed": True}, save)

    def jacobian(oracle, baseline, seconds, save):
        calls.append("jacobian")
        return campaign.previous._saved({"passed": True}, save)

    def coordinate(oracle, baseline, jacobian, index, seconds, save, *, probe_runner):
        calls.append(index)
        probe_runner(
            "probe_00",
            {},
            lambda save: campaign.previous._saved({"index": index}, save),
        )
        return campaign.previous._saved({"passed": True, "index": index}, save)

    monkeypatch.setattr(campaign.nonlinear, "precision_replay", replay)
    monkeypatch.setattr(
        campaign.nonlinear, "precision_baseline", lambda *a: {"passed": True}
    )
    monkeypatch.setattr(campaign.nonlinear, "precision_jacobian", jacobian)
    monkeypatch.setattr(campaign.nonlinear, "precision_coordinate", coordinate)
    monkeypatch.setattr(
        campaign.nonlinear,
        "precision_summary",
        lambda oracle, rows: {
            "checked": [r["index"] for r in rows],
            "search_qualified": False,
        },
    )
    run.precision()
    assert calls == ["BDF", "Radau", "jacobian", 8, 10, 19, 22]
    run.precision()
    assert calls == ["BDF", "Radau", "jacobian", 8, 10, 19, 22]
    ops = campaign._operations(run.folder, run.identity, run.allowances)
    assert not campaign.first._value(ops["precision_summary"])["search_qualified"]
    import shutil

    shutil.rmtree(run.folder / "probes/coordinate_008")
    with pytest.raises(ValueError, match="nested probe"):
        campaign._operations(run.folder, run.identity, run.allowances)


def test_reservations_and_fixed_roster():
    settings = campaign.Policy()
    assert campaign.SELECTED == (3, 10, 12, 14, 15)
    for kind in ("targeted_precision", "nonlinear_search", "timescale_profiles"):
        assert sum(campaign._allowances({"kind": kind}, settings).values()) <= 8820
    with pytest.raises(ValueError):
        campaign.Policy(profile_seconds=121)
    with pytest.raises(ValueError):
        campaign.Policy(search_seconds=1801)


@pytest.mark.parametrize("start_verified", [True, False])
def test_nonlinear_search_requires_its_verified_start_and_resumes_once(
    prepared, monkeypatch, start_verified
):
    plan, data = campaign.verify(prepared)
    run = campaign._Run(prepared, plan, data, 0)
    run.task = run.task | {"kind": "nonlinear_search", "coordinates": 1}
    run.allowances = campaign._allowances(run.task, run.policy)
    run.record = deepcopy(run.record)
    run.record["operations"] = {
        name: {"status": "complete", "value": value}
        for name, value in {
            "baseline": {"residual": [0.0]},
            "jacobian": {},
            "coordinate_000": {"index": 0},
            "qualification_summary": {},
        }.items()
    }
    run.oracle = SimpleNamespace(start={"x": 1.0})
    calls = []
    qualified = {"search_qualified": True, "start_training_nmse": 2.0}

    def authenticate(oracle, baseline, jacobian, directions, summary):
        assert directions[0]["base_residual"] == baseline["residual"]
        return qualified

    def verify(oracle, parameters, seconds, save, *, expected_nmse):
        calls.append(("verify", expected_nmse))
        return campaign.previous._saved(
            {
                "status": "complete",
                "parameters": dict(parameters),
                "training_nmse": expected_nmse,
                "original_equations_verified": start_verified,
            },
            save,
        )

    def search(oracle, start, seconds, maximum_calls, save, *, qualification):
        assert qualification == qualified
        calls.append(("search", None))
        return campaign.previous._saved(
            {
                "best": {"parameters": {"x": 2.0}, "training_nmse": 1.0},
                "last_accepted_parameters": {"x": 3.0},
            },
            save,
        )

    monkeypatch.setattr(campaign.nonlinear, "authenticate_qualification", authenticate)
    monkeypatch.setattr(campaign.nonlinear, "verify_endpoint", verify)
    monkeypatch.setattr(campaign.nonlinear, "search", search)
    monkeypatch.setattr(run, "endpoints", lambda history: None)
    run.search()
    expected = [("verify", 2.0)]
    if start_verified:
        expected += [("search", None), ("verify", 1.0)]
    assert calls == expected
    run.search()
    assert calls == expected
