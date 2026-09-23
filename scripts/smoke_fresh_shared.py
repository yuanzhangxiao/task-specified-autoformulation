#!/usr/bin/env python3
"""Fresh shared construction, two observed targets, response revision and pruning."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import fresh_shared as campaign
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_write
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.schemas.staged_topology import PublicVariable
from autoformalism.search.training_evidence import build_training_evidence

spec = importlib.util.spec_from_file_location(
    "fresh_shared_fixture",
    Path(__file__).with_name("smoke_shared_process_integration.py"),
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def fixture(root: Path) -> dict:
    """Synthetic packed public data; no benchmark trajectories or historical models."""
    plan = old.fixture(root)
    config = io.DeadlineConfig.model_validate(
        {
            **plan["config"],
            "protocol": io.FRESH_PROTOCOL,
            "full_only": True,
            "fit_profile": "collocation-multi-target-v1",
        }
    )
    cell = next(iter(plan["cells"].values()))
    cell["context"]["targets"].append("w")
    cell["brief"]["public_variables"].append(
        PublicVariable(name="w", data_role="target").model_dump(mode="json")
    )
    cell["target_contract"]["targets"].append(
        {"target_channel": "w", "public_requirement": "Generate w causally."}
    )
    for name in ("training", "validation"):
        for row in cell[name]["rows"]:
            t = np.asarray(row["time"])
            baseline = 0.8 * row["external_inputs"]["u01"][0] / 0.3
            initial = 1 + 2 * row["targets"]["v01"][0]
            row["targets"]["w"] = (
                baseline + (initial - baseline) * np.exp(-0.3 * t)
            ).tolist()
    cell["evidence"] = build_training_evidence(
        public.unpack_split(PublicSplit.model_validate(cell["training"])),
        ValidationContext.model_validate(cell["context"]),
    ).model_dump(mode="json")
    plan.pop("artifact_sha256")
    plan.update(
        protocol=io.FRESH_PROTOCOL,
        config=config.model_dump(mode="json"),
        tasks=io.tasks(config),
        launcher_sha256=io.launcher_hash(io.FRESH_PROTOCOL),
        fresh_shared_policy=campaign.policy(),
    )
    (root / "plan.json").unlink()  # Replace only this isolated toy fixture.
    return sealed_write(root / "plan.json", plan)


def transport_for(calls):
    construction = old.transport_for(calls)

    def transport(url, body, timeout):
        text = body["messages"][1]["content"]
        user = json.loads(text if text.startswith("{") else text.split("\n", 1)[1])
        if user.get("selected_lhs", {}).get("name") == "w":
            raw = {
                "terms": [
                    {
                        "sources": ["m"],
                        "outer_weight_sign": "positive",
                        "scientific_role": "observed memory readout",
                    }
                ],
                "inventory_revision": None,
            }
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
                ],
                "usage": {"total_tokens": 100},
            }
        if user.get("protocol") != "response-oriented-revision-1":
            reply = construction(url, body, timeout)
            raw = json.loads(reply["choices"][0]["message"]["content"])
            if "variables" in raw:
                raw["variables"].append(
                    {
                        "name": "w",
                        "definition": "algebraic",
                        "scientific_role": "observed memory readout",
                    }
                )
                reply["choices"][0]["message"]["content"] = json.dumps(raw)
            return reply
        calls.append(user)
        assert set(user["training_evidence"]["target_overview"]) == {"v01", "w"}
        assert user["parameter_declaration_policy"]
        assert user["shared_law_relationships"]
        assert "validation" not in user and "test" not in user
        raw = {
            "hypothesis": "Replace the common saturating memory law with a linear law.",
            "evidence_refs": [user["evidence_catalog"][0]["ref"]],
            "equations": [{"component": "q", "kind": "algebraic", "expression": "m"}],
        }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
            ],
            "usage": {"total_tokens": 100},
        }

    return transport


def run(root: Path) -> dict:
    plan = fixture(root)
    io.verify(root)
    task, calls = plan["tasks"][0], []
    for index in (0, 1):
        client = pipeline._client(
            root,
            plan,
            task,
            index,
            "http://offline",
            lambda: True,
            transport_for(calls),
        )
        client.token_transport = lambda *args: {"count": 4000, "max_model_len": 32768}
        proposal = pipeline.propose_one(root, plan, task, index, client)
        assert proposal["status"] == ("constructed" if index == 0 else "committed"), (
            proposal
        )
        result = pipeline.fit_one(root, plan, task, index)
        assert result["selected"] is not None, result
        assert set(result["trial"]["fit"]["training"]["per_target_normalized_mse"]) == {
            "v01",
            "w",
        }
        if index:
            changed = proposal["decision"]["provenance"]["shared_law_revision"]
            assert next(x for x in changed["after"] if x["process"] == "q")[
                "direct_consumers"
            ] == ["m", "v01"]
        count = len(calls)
        assert proposal == pipeline.propose_one(root, plan, task, index, client)
        assert result == pipeline.fit_one(root, plan, task, index)
        assert len(calls) == count
    pruned = campaign.run_one(root, 0)
    assert pruned["status"] == "complete", pruned
    assert pruned["choice"]["status"] == "ready", pruned["choice"]
    assert all(
        f["profile"] == "collocation-multi-target-v1"
        for f in pruned["fits"].values()
        if f
    )
    before = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    assert campaign.run_one(root, 0) == pruned
    assert before == {str(p): p.read_bytes() for p in root.rglob("*.json")}
    assert campaign.report(root)["status"] == "complete"
    return {
        "status": "passed",
        "observed_targets": 2,
        "real_search_fits": 2,
        "pruning_fit_arms": sorted(pruned["fits"]),
        "live_llm_calls": 0,
        "fresh_shared_construction": True,
        "exact_resume": True,
        "test_data_opened": False,
        "scientific_critic_called": False,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="fresh-shared-smoke-") as directory:
        print(json.dumps(run(Path(directory)), indent=2))
