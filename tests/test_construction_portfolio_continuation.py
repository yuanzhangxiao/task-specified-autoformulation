"""Prerequisite reuse, gated incumbent continuation and durable nested probes."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoformalism.research import construction_portfolio_continuation as campaign
from tests.test_construction_portfolio_fitting import completed  # noqa: F401
from tests.test_construction_portfolio_recovery import (  # noqa: F401
    nonlinear_control,
    recovered,
)
from tests.test_construction_portfolio_recovery import (
    policy as old_policy,
)


def policy():
    return campaign.Policy(
        population="synthetic-smoke",
        prerequisite_seconds=10,
        coordinate_seconds=10,
        qualification_seconds=20,
        search_seconds=15,
        verification_seconds=15,
        diagnostic_seconds=15,
        maximum_calls=40,
        inference=campaign.kernel.NumericalPolicy(
            seconds=10, maximum_calls=40, verification_seconds=10
        ),
    )


@pytest.fixture(scope="module")
def continued(recovered, tmp_path_factory):  # noqa: F811
    source = recovered[0]
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    root = tmp_path_factory.mktemp("portfolio-continuation")
    campaign.prepare(root, source, policy())
    results = [campaign.run_task(root, i) for i in range(2)]
    assert before == {p: p.read_bytes() for p in source.rglob("*.json")}
    return root, results


def test_affine_smoke_preserves_incumbents_and_qualifies_before_search(continued):
    root, results = continued
    plan, data = campaign.verify(root)
    assert campaign.reservation_ceiling(plan) < 13200
    for index, result in enumerate(results):
        assert result["qualification"]["passed"], result
        assert result["shared_parameter_search_attempted"]
        assert result["operation_statuses"]["search"] == "complete"
        assert result["continuation_status"] in {
            "continued_verified",
            "continuation_no_improvement",
        }
        assert result["retained"]["original_equations_verified"]
        assert (
            result["training_reconstruction_nmse"]
            <= data["records"][index]["source_record"]["endpoint"]["training_nmse"]
        )
        assert result["training_prefix_forecast"]["normalized_mse"] < 1e-8
        assert not result["prefix_scores_used_for_selection"]
        assert not result["test_data_opened"]
    assert campaign.report(root) == campaign.report(root)
    assert campaign.report(root)["recorded"] == 2
    assert campaign.inspection(root).is_file()


def test_terminal_resume_is_read_only(continued, monkeypatch):
    root, results = continued
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(
        campaign.polish, "make_oracle", lambda *a: pytest.fail("rebuilt")
    )
    assert [campaign.run_task(root, i) for i in range(2)] == results
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


def test_unqualified_or_worse_endpoint_never_replaces_source():
    def op(value):
        return {"status": "complete", "value": value}

    endpoint = {
        "status": "complete",
        "training_nmse": 0.5,
        "original_equations_verified": True,
    }
    record = {"source_record": {"endpoint": endpoint}}
    operations = {
        "endpoint_verification": op(endpoint | {"training_nmse": 0.2}),
        "start_verification": op(endpoint),
        "qualification": op({"passed": True}),
        "search": op({"best": {"parameters": {}}}),
    }
    assert campaign._retained(record, operations)[0] == "continuation"
    for name in operations:
        fewer = {k: v for k, v in operations.items() if k != name}
        assert campaign._retained(record, fewer) == ("source", endpoint)
    for score, verified in [(0.7, True), (0.2, False)]:
        altered = operations | {
            "endpoint_verification": op(
                endpoint
                | {"training_nmse": score, "original_equations_verified": verified}
            )
        }
        assert campaign._retained(record, altered) == ("source", endpoint)


def test_source_and_frozen_roster_integrity(recovered, tmp_path, monkeypatch):  # noqa: F811
    import shutil

    source = tmp_path / "source"
    shutil.copytree(recovered[0], source)
    with pytest.raises(ValueError, match="separate"):
        campaign.prepare(source / "nested", source, policy())
    data = campaign._source_inputs(source, policy())
    changed = deepcopy(data)
    changed["records"].pop()
    with pytest.raises(ValueError, match="roster"):
        campaign.prepare_inputs(tmp_path / "bad", changed, policy())
    campaign.prepare_inputs(tmp_path / "new", data, policy())
    monkeypatch.setattr(campaign.public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="source/runtime"):
        campaign.verify(tmp_path / "new")
    assert campaign.verify(tmp_path / "new", runtime=False)
    terminal = source / "results/candidate-00/result.json"
    terminal.unlink()
    with pytest.raises(FileNotFoundError):
        campaign.prepare(tmp_path / "incomplete", source, policy())


def test_production_roster_and_limits():
    settings = campaign.Policy()
    for coordinates in (28, 29):
        task = {"kind": "nonlinear_prerequisites", "coordinates": coordinates}
        assert sum(campaign._allowances(task, settings, 16).values()) <= 9600
    assert (
        sum(
            campaign._allowances({"kind": "affine_continuation"}, settings, 16).values()
        )
        == 8400
    )
    assert campaign.SELECTED == (3, 10, 12, 14, 15)


def nested_receipts(folder):
    identity = {"test": "nested"}
    allowances = {"coordinate_000": 300}

    def parent(save):
        child = campaign.journal._operation(
            folder / "probes/coordinate_000",
            identity,
            "probe_00",
            120,
            lambda save: campaign._saved({"primitive": True}, save),
        )
        return campaign._saved(
            {"primitive_receipts": {"probe_00": campaign.public.content_sha256(child)}},
            save,
        )

    campaign.journal._operation(folder, identity, "coordinate_000", 300, parent)
    return identity, allowances


@pytest.mark.parametrize(
    "removed", ["probes", "probes/coordinate_000", "operations/coordinate_000"]
)
def test_nested_probe_deletion_cannot_hide_work(tmp_path, removed):
    import shutil

    identity, allowances = nested_receipts(tmp_path)
    original = campaign._operations(tmp_path, identity, allowances)
    assert len(original) == 1  # Inclusive parent accounting never counts child twice.
    shutil.rmtree(tmp_path / removed)
    with pytest.raises(ValueError, match="nested probe"):
        campaign._operations(tmp_path, identity, allowances)


def test_probe_tampering_and_unknown_parent_rejected(tmp_path):
    identity, allowances = nested_receipts(tmp_path)
    path = tmp_path / "probes/coordinate_000/operations/probe_00/result.json"
    value = campaign.public._read(path)
    value["value"]["value"]["primitive"] = False
    campaign.public._write(path, value)
    with pytest.raises(ValueError):
        campaign._operations(tmp_path, identity, allowances)


def new_nonlinear(recovered, tmp_path, monkeypatch):  # noqa: F811
    """Real source journals with cheap deterministic orchestration primitives."""
    data, _ = nonlinear_control(recovered, monkeypatch)
    source = tmp_path / "source"
    campaign.previous.prepare_inputs(source, data, old_policy())
    campaign.previous.run_task(source, 0)
    root = tmp_path / "new"
    campaign.prepare(root, source, policy())
    oracle = SimpleNamespace(
        names=("a", "b"),
        original=SimpleNamespace(names=("a", "b")),
        portfolio_audit={"route": "original_ODE"},
    )
    monkeypatch.setattr(campaign.prerequisites, "make_oracle", lambda *a: oracle)
    calls = []

    def primitive(name):
        def run(*args):
            calls.append(name)
            return campaign._saved({"passed": True}, args[-1])

        return run

    monkeypatch.setattr(
        campaign.prerequisites, "import_baseline", primitive("baseline")
    )
    monkeypatch.setattr(
        campaign.prerequisites, "prepare_jacobian", primitive("jacobian")
    )

    def coordinate(oracle, baseline, jacobian, index, seconds, save, *, probe_runner):
        calls.append(index)
        probe_runner(
            "probe_00", {}, lambda save: campaign._saved({"index": index}, save)
        )
        return campaign._saved({"passed": True, "index": index}, save)

    monkeypatch.setattr(campaign.prerequisites, "qualify_coordinate", coordinate)
    monkeypatch.setattr(
        campaign.prerequisites,
        "aggregate_qualification",
        lambda o, r: {
            "search_qualified": len(r) == 2 and all(x.get("passed") for x in r)
        },
    )
    return root, calls


def test_shared_prerequisites_run_once_and_probe_receipts_are_authenticated(
    recovered,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    root, calls = new_nonlinear(recovered, tmp_path, monkeypatch)
    result = campaign.run_task(root, 0)
    assert calls == ["baseline", "jacobian", 0, 1]
    assert result["continuation_status"] == "qualified_for_search"
    assert not result["shared_parameter_search_attempted"]
    assert campaign.run_task(root, 0) == result
    assert calls == ["baseline", "jacobian", 0, 1]
    assert campaign.report(root)["rows"][0] == result


@pytest.mark.parametrize("stage", ["import_baseline", "prepare_jacobian"])
def test_failed_prerequisite_blocks_all_probes(recovered, tmp_path, monkeypatch, stage):  # noqa: F811
    root, calls = new_nonlinear(recovered, tmp_path, monkeypatch)
    monkeypatch.setattr(
        campaign.prerequisites,
        stage,
        lambda *a: campaign._saved({"passed": False}, a[-1]),
    )
    result = campaign.run_task(root, 0)
    assert not any(type(x) is int for x in calls)
    assert result["continuation_status"] == "prerequisite_unavailable"


def test_interrupted_coordinate_does_not_repeat_prerequisites_or_probe(
    recovered,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    root, calls = new_nonlinear(recovered, tmp_path, monkeypatch)
    coordinate = campaign.prerequisites.qualify_coordinate

    def interrupt(*a, **kw):
        value = coordinate(*a, **kw)
        if a[3] == 0:
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr(campaign.prerequisites, "qualify_coordinate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(root, 0)
    assert calls == ["baseline", "jacobian", 0]
    result = campaign.run_task(root, 0)
    assert calls == ["baseline", "jacobian", 0, 1]
    assert result["continuation_status"] == "derivatives_unqualified"
    assert not result["cost"]["accounting_complete"]
    assert campaign.report(root)["rows"][0] == result
