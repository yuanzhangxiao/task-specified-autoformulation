"""Screened selection, topology checks, resume, and ACES submission."""

import json
import tarfile

import pytest
from pydantic import ValidationError

from autoformalism.research import preproposal_cases as cases
from autoformalism.research import preproposal_selection as selection
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.staged_topology import content_hash
from scripts import preproposal_selection_20b as cli
from scripts import submit_t2_breadth as submitter
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_preproposal_screen import complementary
from tests.test_topology_repair_generator import evidence


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    channels = [
        {
            "name": v,
            "data_role": "target" if v in ("level", "upstream") else "external_input",
        }
        for v in ("level", "upstream", "rain", "spill")
    ]
    brief = PublicScientificBrief(
        scientific_context="Level responds to reservoir flows.",
        public_variables=channels,
        requirements=[],
    )
    frozen = {
        "case_id": "reservoir",
        "target": "level",
        "inventory": [
            {"name": n, "kind": "differential"} for n in ("level", "upstream")
        ],
        "public_brief": brief.model_dump(mode="json"),
        "evidence": evidence(complementary()).model_dump(mode="json"),
        "histories": ["rain", "spill"],
        "initials": ["upstream"],
        "required_histories": [],
        "provenance": {},
    }
    baseline_only = dict(
        frozen, case_id="no_extra", required_histories=["rain", "spill"]
    )
    raw = {r["case_id"]: r for r in (frozen, baseline_only)}
    monkeypatch.setattr(
        cases, "PUBLIC_DIGESTS", {k: content_hash(r) for k, r in raw.items()}
    )
    payload = {
        "selection": {"cases": [cases.selection_case(r) for r in raw.values()]},
        "frozen_cases": raw,
    }
    source = tmp_path / "input.json"
    source.write_text(json.dumps(payload))
    root = tmp_path / "result"
    cli.prepare(root, source)
    return root, source


def test_validation_and_skipping(prepared):
    root, source = prepared
    preview = cli.preview(root)
    assert preview[1]["status"] == "no_added_information_required"
    assert preview[0]["user"]["target"] == "level"
    assert "trajectories" not in preview[0]["user"]
    assert all(c["sources"] for c in preview[0]["user"]["candidates"])
    assert len(preview[0]["user"]["candidates"]) == 2
    payload = json.loads(source.read_text())
    c = selection.SelectionCase.model_validate(payload["selection"]["cases"][0])
    schema = selection.selection_schema(c)
    chosen = {
        "candidate_id": c.candidates[0].candidate_id,
        "rationale": "Reservoir flow.",
    }
    assert (
        selection.assess(c, schema(selections=[chosen, chosen]))["status"]
        == "invalid_selection"
    )
    with pytest.raises(ValidationError):
        schema(selections=[dict(chosen, candidate_id="arbitrary")])
    payload["frozen_cases"]["reservoir"]["target"] = "upstream"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differ"):
        cli.input_payload(source)


def test_selections_separate_realized_graph_failures_and_resume(prepared):
    root, _ = prepared
    requests = []
    menu = cli.preview(root)[0]["user"]["candidates"]

    submitted_seeds = {
        int(content_hash([seed, suffix, 0])[:8], 16) % (2**31): seed
        for seed in cli.SEEDS
        for suffix in ("select", "topology-1", "topology-2")
    }

    def generate(url, body, timeout):
        req = json.loads(body["messages"][1]["content"])
        requests.append(body)
        seed = submitted_seeds[body["seed"]]
        if "candidates" in req:
            if seed == 100:
                raise TimeoutError("no response")
            items = [
                {
                    "candidate_id": c["candidate_id"],
                    "rationale": "Forcing can change level.",
                }
                for c in menu
            ]
            if seed == 101:
                return reply({"selections": [items[0], items[0]]})
            return reply({"selections": items if seed == 102 else items[:1]})
        graph = {"level": ["upstream", "rain", "spill"], "upstream": ["spill"]}
        if seed == 103:
            graph["level"] = ["level"]
        return reply(
            {
                "hypothesis": "Flow changes storage.",
                "sources": graph,
                "latent_initializers": [],
            }
        )

    def run():
        return cli.run(
            root, "http://127.0.0.1", transport=generate, token_transport=tokenizer
        )

    result = run()
    assert result["maximum_generation_requests"] == 15
    assert result["completed_active_draws"] == 5 and result["status"] == "complete"
    assert len(requests) == result["physical_generation_requests"] == 9
    assert result["unknown_usage"] == 1 and result["optimizer_calls"] == 0
    r, skipped = result["cases"]
    assert r["selection_status_counts"] == {
        "provider_failure": 1,
        "invalid_selection": 1,
        "selected": 3,
    }
    assert r["topology_status_counts"] == {
        "topology_screen_passed": 3,
        "training_conflict": 1,
    }
    assert r["first_choice_frequencies"] == {menu[0]["candidate_id"]: 3}
    assert r["draws_with_passing_topology"] == 2
    assert not skipped["draws"]
    assert run() == result and len(requests) == 9
    with tarfile.open(cli.inspect(root)["archive"]) as tar:
        assert json.load(tar.extractfile("summary.json"))["cases"] == result["cases"]
    receipt = next(root.glob("steps/*--topology-1/calls/*.json"))
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match="changed provider receipt"):
        cli.status(root)


def test_five_draws_submit_new_20b_worker_with_its_own_identity(
    prepared, tmp_path, monkeypatch
):
    root, _ = prepared
    image = tmp_path / "vllm.sif"
    image.write_bytes(b"mock")
    monkeypatch.setattr(submitter.aces, "_assets", lambda size: (image, {"size": size}))
    monkeypatch.setattr(submitter.aces.harmony_vocab, "verify", lambda _: {})
    calls = []

    def submit(path, argv):
        calls.append(argv)
        return "12345"

    monkeypatch.setattr(submitter, "_submit_job", submit)
    assert submitter.submit(root) == submitter.submit(root)
    assert len(calls) == 1 and calls[0][-1] == "preproposal-selection"
    assert {
        "--gres=gpu:h100:1",
        "--time=01:00:00",
        "--job-name=phasec-preselect-20b",
    } <= set(calls[0])
    assert (
        submitter.verify(root, check_assets=True)["maximum_generation_requests"] == 15
    )


def test_two_arms_match_sampler_seeds_and_keep_separate_receipts(
    prepared, tmp_path, monkeypatch
):
    _, source = prepared
    payload = json.loads(source.read_text())
    twin = dict(payload["frozen_cases"]["reservoir"], case_id="reservoir_twin")
    payload["frozen_cases"][twin["case_id"]] = twin
    payload["selection"]["cases"].append(cases.selection_case(twin))
    monkeypatch.setattr(
        cases,
        "PUBLIC_DIGESTS",
        {k: content_hash(v) for k, v in payload["frozen_cases"].items()},
    )
    source.write_text(json.dumps(payload))
    root = tmp_path / "paired"
    cli.prepare(root, source)
    checks = cli.preflight(root)
    assert checks["maximum_selection_requests"] == 10
    assert checks["maximum_topology_requests"] == 20
    assert checks["generation_requests_started"] == 0
    sampler_seeds = []

    def transport(url, body, timeout):
        sampler_seeds.append(body["seed"])
        req = json.loads(body["messages"][1]["content"])
        if "candidates" in req:
            return reply(
                {
                    "selections": [
                        {
                            "candidate_id": req["candidates"][0]["candidate_id"],
                            "rationale": "A plausible flow.",
                        }
                    ]
                }
            )
        return reply(
            {
                "hypothesis": "Flow changes level.",
                "sources": {
                    "level": ["rain", "spill", "upstream"],
                    "upstream": ["spill"],
                },
                "latent_initializers": [],
            }
        )

    summary = cli.run(
        root, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert summary["physical_generation_requests"] == 20
    assert sampler_seeds[:10] == sampler_seeds[10:]
    assert len(set(sampler_seeds)) == 10
