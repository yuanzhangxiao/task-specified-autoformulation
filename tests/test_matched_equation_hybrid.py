"""Check the training-only hybrid packet and exactly-once provider receipts."""

import json

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import matched_equation_hybrid as hybrid
from scripts import matched_equation_hybrid_provider as worker


def packet(label):
    """Two admissible formula options with a single observed forcing."""

    def formula(rhs):
        return {"equations": {"y": rhs}, "initializers": {}, "parameters": {}}

    return {
        "panel_id": label,
        "public_task": "Predict an observed response to a forcing.",
        "train_rows": [
            {
                "trajectory_id": "train_0",
                "time": [0, 1, 2],
                "targets": {"y": [1, 2, 3]},
                "inputs": {"u": [3, 2, 1]},
            }
        ],
        "graph": {
            "variable_types": {"y": "differential"},
            "initializers": {"y": {"mode": "observed_initial", "channel": "y"}},
            "sources": {"y": ["u"]},
        },
        "shape": {
            "candidates": [formula("u"), formula("u * u")],
            "evidence": {"split": "train", "events": ["train_0"], "equations": {}},
        },
    }


def test_strict_prompt_and_fold_boundary():
    item = packet("graph00-fold0")
    item["held_observations"] = {"withheld_marker": [37]}
    request = hybrid.make_body(item)
    assert "withheld_marker" not in json.dumps(request)
    rule = request["response_format"]["json_schema"]["schema"]["$defs"]["SelectFamily"]
    assert rule["properties"]["shortlist_id"]["enum"] == ["B"]
    item["shape"]["evidence"]["events"] += ["train_wrong"]
    with pytest.raises(ValueError, match="fitting events"):
        hybrid.make_body(item)


def test_cached_collection_and_invalid_replies(tmp_path):
    labels = sorted(f"graph{n:02}-fold{f}" for n in range(6) for f in range(2))
    for name in labels:
        sealed_write(tmp_path / "packets" / f"{name}.json", packet(name))
    manifest = hybrid.prepare(tmp_path)
    assert len(manifest["requests"]) == 12
    assert hybrid.result_for(packet(labels[0]), tmp_path)["status"] == "pending"
    assert hybrid.prepare(tmp_path) == manifest
    counter = 0

    def generate(url, request):
        nonlocal counter
        counter += 1
        if counter == 4:
            raise TimeoutError("server timeout")
        content = {"candidate": {"shortlist_id": "B"}}
        if counter == 1:
            content = {"candidate": {"shortlist_id": "A"}}
        if counter == 2:
            content = {
                "candidate": {
                    "equations": {"y": "u + hidden"},
                    "initializers": {},
                    "parameters": [],
                }
            }
        return {
            "model": worker.MODEL,
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(content)}}
            ],
        }

    folder = tmp_path / "hybrid"
    worker.run(folder, "http://127.0.0.1:1", generate)
    worker.run(folder, "http://127.0.0.1:1", lambda *_: pytest.fail("Duplicate call"))
    assert counter == 12
    rows = [hybrid.result_for(packet(label), tmp_path) for label in labels]
    assert rows[0]["status"] == rows[1]["status"] == "invalid"
    assert rows[2]["status"] == "eligible" and rows[2]["action"] == "selected_B"
    assert rows[3]["status"] == "provider_error"
    path = folder / "generate-responses" / f"{labels[-1]}.json"
    corrupted = json.loads(path.read_text())
    corrupted["response"]["choices"][0]["finish_reason"] = "length"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="modified"):
        hybrid.result_for(packet(labels[-1]), tmp_path)


def test_rejects_unfinished_input_or_remote_endpoint(tmp_path):
    with pytest.raises(ValueError, match="Expected 12"):
        hybrid.prepare(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        worker.run(tmp_path, "https://unapproved.example", lambda *_: None)
