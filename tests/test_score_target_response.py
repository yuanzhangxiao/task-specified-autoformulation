"""Checkpoint folders of a scored suite root, as the suite report reads them."""

from __future__ import annotations

from pathlib import Path

from scripts import score_target_response as target


def test_baseline_and_study_subjects_have_separate_folders():
    plan = {
        "rows": [{"source_id": "d3:0"}, {"source_id": "d3:1"}],
        "study_rows": [{"source_id": "full:0"}],
    }
    folders = target.subject_folders(Path("root"), plan)
    assert folders == {
        "d3:0": Path("root/subjects/0000"),
        "d3:1": Path("root/subjects/0001"),
        "full:0": Path("root/study/subjects/0000"),
    }
