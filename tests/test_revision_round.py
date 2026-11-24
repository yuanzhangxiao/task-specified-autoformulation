"""Check paired selectors, receipts, and the train/development boundary."""

import json
from types import SimpleNamespace

import pytest

from scripts import revision_round as study
from scripts import revision_round_provider as worker
from scripts import revision_round_requests as requests


def fake_packet():
    return {
        "parent": {"power": 1},
        "evidence": "observed trends",
        "algorithm_edits": [{"power": 1.5}, {"power": 2}, {"power": 3}],
        "edit_contract": "one power",
        "edit_schema": {"type": "object", "properties": {"power": {"type": "number"}}},
    }


def complete(root, stage, build):
    manifest = worker.load_stage(root / f"{stage}.json")
    for label, body in manifest["requests"].items():
        worker.save(
            root / f"{stage}-responses" / f"{label}.json",
            {
                "request_sha256": worker.digest(body),
                "status": "responded",
                "response": {
                    "choices": [{"message": {"content": json.dumps(build(label))}}]
                },
            },
        )


def test_pipeline_with_duplicate_proposals_and_failed_judges(tmp_path, monkeypatch):
    fits, dev_calls = [], []

    def score(edit):
        fits.append(edit)
        return {"eligible": True, "train_nmse": abs(edit["power"] - 1.5)}

    def validate(edit, fitted):
        assert (tmp_path / "frozen.json").exists()
        dev_calls.append(edit)
        return {"nmse": fitted["train_nmse"]}

    def screen():
        return {"edits": fake_packet()["algorithm_edits"], "screened": 3}

    model = SimpleNamespace(
        packet=fake_packet, score=score, validate=validate, screen=screen
    )
    monkeypatch.setattr(study, "adapter", lambda _: model)
    monkeypatch.setattr(study, "fit_execution", lambda *_: "fit-v1")
    study.prepare(tmp_path)
    study.prepare_screens(tmp_path)
    complete(
        tmp_path,
        "generate",
        lambda _: {
            "alternatives": [
                {"power": 1.0},
                {"power": 1.5},
                {"power": 1.5},
            ]
        },
    )
    training = study.shortlist(tmp_path)
    stage = worker.load_stage(tmp_path / "rank.json")
    for req in stage["requests"].values():
        user = json.loads(req["messages"][1]["content"])
        assert set(user) == {"edit_contract", "parent", "evidence", "options"}
    assert not dev_calls
    assert len(fits) == 12
    study.shortlist(tmp_path)
    assert len(fits) == 12
    complete(
        tmp_path,
        "rank",
        lambda label: {
            "rank": []
            if label.endswith("101")
            else list(training["label_maps"][label]),
        },
    )
    decisions = study.freeze(tmp_path)["choices"]["insulin"]
    assert decisions["20b_all_nine"]["eligible_slots"] == 9
    assert decisions["20b_all_nine"]["distinct_eligible_edits"] == 1
    assert decisions["20b_seed102_three"]["slots"] == 3
    assert decisions["algorithm"]["20b"]["101"] is None
    assert decisions["algorithm"]["algorithm"] == study.identity({"power": 1.5})
    a = study.validate(tmp_path)
    count = len(dev_calls)
    assert count > 0
    assert study.validate(tmp_path) == a and len(dev_calls) == count
    monkeypatch.setattr(study, "fit_execution", lambda *_: "fit-v2")
    with pytest.raises(ValueError, match="Cached edit"):
        study.shortlist(tmp_path)
    with pytest.raises(ValueError, match="executable"):
        study.validate(tmp_path)


def test_request_blinding_and_incomplete_provider(tmp_path):
    packet = fake_packet()
    req = requests.generation(packet, 100)
    assert "algorithm_edits" not in req["messages"][1]["content"]
    path = tmp_path / "generate-responses" / "reactor-100.json"
    base = {"request_sha256": worker.digest(req), "status": "reserved"}
    worker.save(path, base)
    with pytest.raises(RuntimeError, match="reservation"):
        study.reply(tmp_path, "generate", "reactor-100", req)
    for bad in ("{", '{"alternatives":[NaN]}', '{"p":1e999}'):
        worker.save(
            path,
            {
                **base,
                "status": "responded",
                "response": {"choices": [{"message": {"content": bad}}]},
            },
        )
        assert study.reply(tmp_path, "generate", "reactor-100", req)[0] is None
    with pytest.raises(ValueError, match="differs"):
        study.reply(tmp_path, "generate", "reactor-100", {})


def test_order_seeded_permutation_and_provider_resume(tmp_path):
    choices = {"k1": {"power": 1}, "k2": {"power": 2}}
    req, labels = requests.ranking(fake_packet(), choices, "reactor", 100)
    assert set(labels.values()) == set(choices)
    assert (
        req
        == requests.ranking(
            fake_packet(), dict(reversed(choices.items())), "reactor", 100
        )[0]
    )
    payloads = {
        f"{c}-{s}": requests.generation(fake_packet(), s)
        for c in worker.CASES
        for s in worker.SEEDS
    }
    study.write_stage(tmp_path, "generate", payloads)
    for name, payload in payloads.items():
        worker.save(
            tmp_path / "generate-responses" / f"{name}.json",
            {
                "request_sha256": worker.digest(payload),
                "status": "reserved",
            },
        )
    worker.run(tmp_path, "generate", "http://127.0.0.1:1")
    assert all(
        json.loads(p.read_text())["status"] == "reserved"
        for p in (tmp_path / "generate-responses").glob("*.json")
    )
