"""Exercise the complete three-slot fit, its held-out boundary and scheduler limits."""

from scripts import basin_process_proposals as proposer
from scripts import fit_basin_process_proposals as scoring
from scripts import submit_t2_breadth as launcher


def test_refits_guard_at_crest_and_marks_lost_transfer():
    cand = {
        "authored_factors": {"upstream_factor": "1/k", "downstream_factor": "k"},
        "revised_graph": {"sources": {"h_down": ["h_up"]}},
    }
    result = {
        "status": "fitted",
        "training_nmse": 1.0,
        "parameters": {"k": 0.0},
        "effective_sources": {"h_down": []},
        "negative_samples": {"h_down": 0},
    }
    issues = scoring.fitted_checks(cand, result)
    assert any("zero head" in issue for issue in issues)
    assert any("lost fitted" in issue for issue in issues)


def test_three_proposals_use_one_h100_for_twenty_minutes(monkeypatch):
    monkeypatch.setattr(launcher.aces.harmony_vocab, "verify", lambda path: {})
    plan = {
        "artifact_sha256": "example",
        "protocol": proposer.PROTOCOL,
        "settings": {"max_output_tokens": 6144},
        "maximum_generation_requests": 3,
        "entrypoints": {},
    }
    record = launcher.execution(plan, "image", {})
    assert launcher._trial(proposer.PROTOCOL) is proposer
    assert (record["model_size"], record["tensor_parallel_size"]) == ("20b", 1)
    assert launcher.time_limit(record["scheduler_walltime_seconds"]) == "00:20:00"
    assert record["maximum_generation_requests"] == 3
