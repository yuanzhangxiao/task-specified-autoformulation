"""Submission, preview, and resume checks for the matched ACES delivery."""

import json
import shlex
from datetime import UTC, datetime

import pytest

from scripts import preproposal_selection_20b as cli
from scripts import review_preproposal_pilot as review
from scripts import submit_t2_breadth as submitter
from scripts import t2_breadth_scheduler as scheduler
from tests import test_preproposal_selection_20b

prepared = test_preproposal_selection_20b.prepared


def test_matched_submission_is_reconciled_and_preview_uses_prepared_requests(
    prepared, tmp_path, monkeypatch
):
    original, source = prepared
    root = tmp_path / "matched"
    cli.prepare(root, source, matched=True)
    image = tmp_path / "vllm.sif"
    image.write_bytes(b"mock")
    monkeypatch.setattr(submitter.aces, "_assets", lambda size: (image, {"size": size}))
    monkeypatch.setattr(submitter.aces.harmony_vocab, "verify", lambda _: {})
    argv = []

    def timeout(wave, command):
        argv.extend(command)
        (wave / "propose.intent.json").write_text(json.dumps({"argv": command}))
        raise ValueError("Socket timed out")

    def lookup(command):
        if command[0] != "sacct":
            return ""
        stamp = datetime.now(UTC).isoformat()
        return "|".join(
            (
                "12345",
                scheduler.getpass.getuser(),
                "phasec-matched-20b",
                "PENDING",
                stamp,
                shlex.join(argv),
            )
        )

    monkeypatch.setattr(submitter, "_submit_job", timeout)
    monkeypatch.setattr(scheduler, "_query", lookup)
    assert submitter.submit(root)["job"] == "12345"
    assert submitter.submit(root)["job"] == "12345"
    assert argv.count("sbatch") == 1
    assert {
        "--gres=gpu:h100:1",
        "--time=01:00:00",
        "--job-name=phasec-matched-20b",
    } <= set(argv)
    assert argv[-1] == "preproposal-selection"
    binding = submitter.verify(root, check_assets=True)
    assert binding["protocol"] == "preproposal-matched-selection-aces-2"
    out = tmp_path / "preview"
    html = review.render_matched(root, out).read_text()
    assert "Graph before repair" in html and "reservoir" in html
    assert "no_added_information_required" in html
    with pytest.raises(ValueError, match="matched selection plan"):
        review.render_matched(original, out)
