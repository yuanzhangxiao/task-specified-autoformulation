"""Evidence isolation, frozen rosters, prompt projection and cached campaign replay."""

import json
from copy import deepcopy

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_comparison as campaign
from autoformalism.research import construction_focus_review as review
from autoformalism.search import construction_evidence as prompts
from autoformalism.search import construction_evidence_repair as repaired_prompts
from autoformalism.search.topology_compatibility import EvidenceContract, build_evidence
from scripts import jetstream_construction_campaign as jetstream
from tests.test_construction_comparison import source_fixture
from tests.test_construction_ledger import equation, variable
from tests.test_explicit_variable_bindings import response
from tests.test_jetstream_construction_campaign import fixture as jetstream_fixture
from tests.test_phase_c_construction_baseline import tokenize

STUDY = "evidence_feedback_confirmation"


def training_bundle(source, output):
    """Build explicit synthetic complete training evidence for a source fixture."""
    plan = sealed_read(source / "plan.json")
    cells = {}
    source_files = {}
    for name, cell in plan["cells"].items():
        context = ValidationContext.model_validate(cell["context"])
        item = Trajectory(
            trajectory_id="training-event",
            time=np.asarray([0.0, 1.0, 2.0]),
            targets={key: np.asarray([0.0, 1.0, 2.0]) for key in context.targets},
            auxiliaries={key: np.ones(3) for key in context.auxiliaries},
            external_inputs={key: np.zeros(3) for key in context.external_inputs},
            fixed_covariates=dict.fromkeys(context.fixed_covariates, 1.0),
            derivatives={},
        )
        cells[name] = build_evidence(
            DatasetSplit(SplitName.TRAIN, (item,), "synthetic-public"),
            context,
            EvidenceContract(
                provenance="Synthetic integration fixture; unknown assumptions"
            ),
        ).model_dump(mode="json")
        source_files[name] = {
            "historical_source_sha256": cell["evidence"]["source_sha256"]
        }
    return sealed_write(
        output,
        {
            "protocol": "construction-training-evidence-bundle-1",
            "source_plan_sha256": plan["artifact_sha256"],
            "cells": cells,
            "source_files": source_files,
            "test_data_opened": False,
        },
    )


def test_freeze_requires_explicit_complete_training_and_preserves_old_policy(tmp_path):
    source_fixture(tmp_path)
    source, root = tmp_path / "new", tmp_path / "study"
    path = tmp_path / "complete.json"
    bundle = training_bundle(source, path)
    with pytest.raises(ValueError, match="explicit training"):
        campaign.freeze(source, root, study=STUDY)
    with pytest.raises(ValueError, match="explicit training"):
        campaign.freeze(
            source,
            root,
            study="mechanism_realization_confirmation",
            training_evidence=path,
        )
    plan = campaign.freeze(source, root, study=STUDY, training_evidence=path)
    assert len(plan["tasks"]) == 40
    assert plan["sampling_seeds"] == [0, 1, 2, 3, 4]
    assert all(t["arm"] == "full" for t in plan["tasks"])
    assert all(
        t["task_id"].endswith("_minimal_evidence_feedback") for t in plan["tasks"]
    )
    assert plan["training_evidence_bundle_sha256"] == bundle["artifact_sha256"]
    assert all(
        c["training_evidence"] == bundle["cells"][n] for n, c in plan["cells"].items()
    )
    assert campaign.freeze(source, root, study=STUDY, training_evidence=path) == plan
    control = campaign.freeze(
        source, tmp_path / "control", study="mechanism_realization_confirmation"
    )
    assert (
        plan["config"] | {"bookkeeping_policy": control["config"]["bookkeeping_policy"]}
        == control["config"]
    )
    assert plan["config"]["model_settings"]["maximum_requests"] == 133
    assert plan["config"]["model_settings"]["maximum_total_tokens"] == 655360
    assert plan["config"]["repair_requests"] == 10
    assert plan["config"]["repair_tokens"] == 262144
    assert all(
        "training_evidence" not in c and "boundary_contract" not in c
        for c in control["cells"].values()
    )
    # Resume uses the fully embedded evidence, without reopening the delivery file.
    path.unlink()
    assert campaign.verify(root) == plan


@pytest.mark.parametrize(
    "corruption", ["heldout", "missing_cell", "context", "digest", "source"]
)
def test_freeze_rejects_wrong_split_roster_context_and_content(tmp_path, corruption):
    source_fixture(tmp_path)
    source, path = tmp_path / "new", tmp_path / "complete.json"
    bundle = training_bundle(source, path)
    raw = deepcopy(bundle)
    name = next(iter(raw["cells"]))
    if corruption == "heldout":
        raw["cells"][name]["split"] = "test"
    elif corruption == "missing_cell":
        raw["cells"].pop(name)
    elif corruption == "context":
        raw["cells"][name]["context"]["targets"] = ["unavailable"]
    elif corruption == "digest":
        raw["cells"][name]["trajectories"][0]["time"][1] = 0.5
    else:
        raw["source_files"][name]["historical_source_sha256"] = "0" * 64
    raw.pop("artifact_sha256")
    path.unlink()
    sealed_write(path, raw)
    with pytest.raises(ValueError):
        campaign.freeze(source, tmp_path / "study", study=STUDY, training_evidence=path)
    assert not (tmp_path / "study/plan.json").exists()


@pytest.mark.parametrize(
    "study,policy",
    [
        (STUDY, prompts.POLICY),
        ("evidence_repair_confirmation", repaired_prompts.POLICY),
    ],
)
def test_jetstream_new_study_default_roster_and_explicit_evidence(
    tmp_path, study, policy
):
    args = jetstream_fixture(tmp_path)
    path = tmp_path / "training.json"
    training_bundle(args[0], path)
    plan = jetstream.prepare(*args, study=study, training_evidence=path)
    assert len(plan["tasks"]) == 40
    assert plan["config"]["bookkeeping_policy"] == policy
    jetstream.validate_settings(plan["config"])
    assert jetstream.verify(args[-1]) == plan
    assert jetstream.prepare(*args, study=study, training_evidence=path) == plan


@pytest.mark.parametrize(
    "study,policy",
    [
        (STUDY, prompts.POLICY),
        ("evidence_repair_confirmation", repaired_prompts.POLICY),
    ],
)
def test_mocked_all_cases_reports_preview_and_cached_resume(tmp_path, study, policy):
    source_fixture(tmp_path)
    source, root = tmp_path / "new", tmp_path / "study"
    path = tmp_path / "training.json"
    training_bundle(source, path)
    plan = campaign.freeze(source, root, study=study, seeds=[0], training_evidence=path)
    bodies = []

    def send(url, body, timeout):
        bodies.append(body)
        packet = json.loads(body["messages"][1]["content"])
        assert "Training highlights:" in packet["training_highlights"]
        assert "complete-training-compatibility-evidence-1" not in json.dumps(packet)
        targets = [
            v["name"]
            for v in packet["public_brief"]["public_variables"]
            if v["data_role"] == "target"
        ]
        driver = next(
            v["name"]
            for v in packet["public_brief"]["public_variables"]
            if v["data_role"] == "external_input"
        )
        edit = {**packet["response_template"], "stage_complete": True}
        if packet["stage"] == "variables":
            edit["variables"] = [variable(name) for name in [*targets, "m"]]
            edit["mechanism_realizations"] = [
                {
                    "requirement_id": r["id"],
                    "components": ["m", *r["targets"]],
                    "memory_states": ["m"] if r["requires_dynamic_memory"] else [],
                }
                for r in packet["public_brief"]["requirements"]
            ]
        elif packet["stage"] in {"equations", "repair"}:
            edit["equations"] = [equation(name, "m", name) for name in targets] + [
                equation("m", driver, "m")
            ]
            for obligation in packet.get("required_external_effects", []):
                target = obligation["target"]
                target_equation = next(
                    e for e in edit["equations"] if e["name"] == target
                )
                target_equation["terms"].append(
                    {
                        "sources": [target],
                        "outer_weight_sign": "negative",
                        "scientific_role": "external discharge",
                    }
                )
                edit["external_exchanges"].append(
                    {
                        "obligation_id": obligation["id"],
                        "receiver": target,
                        "local_term_index": 1,
                        "shared_process": None,
                        "explanation": "Discharge leaves the modeled system.",
                    }
                )
        return response(edit)

    for task in plan["tasks"]:
        result = campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
        assert result["status"] == "topology_complete", result["assessment"]["errors"]
    campaign.report(root, plan)
    count = len(bodies)
    for task in plan["tasks"]:
        campaign.propose(
            root, plan, task, "http://offline", transport=send, token_transport=tokenize
        )
    assert len(bodies) == count
    task = plan["tasks"][0]
    event = sealed_read(
        root / "results" / task["task_id"] / "construction/events/000.json"
    )
    projected = review.preview(
        plan["cells"][task["benchmark_id"]], task, event, bodies[0], policy
    )
    assert projected["after"] == bodies[0]
    path.unlink()
    summary = review.render(root, tmp_path / "review", policy)
    assert summary["requests"] == count and summary["llm_calls"] == 0
    assert summary["new_schedule"]["maximum_global_repair_calls"] == 10
    text = (tmp_path / "review/PROMPTS.html").read_text()
    assert "Historical replies are not answers" in text
    assert "compatibility_decisions" in text and "external_exchanges" in text
    assert "output JSON schema is unchanged" not in text


def test_old_source_preview_requires_explicit_evidence_and_leaves_source_unchanged(
    tmp_path,
):
    source_fixture(tmp_path)
    source, root = tmp_path / "new", tmp_path / "historical"
    path = tmp_path / "training.json"
    training_bundle(source, path)
    campaign.freeze(source, root, study="mechanism_realization_confirmation", seeds=[0])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="explicit training"):
        review.inspect(root, prompts.POLICY)
    summary, rows = review.inspect(root, prompts.POLICY, training_evidence=path)
    assert rows == [] and summary["llm_calls"] == 0
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
