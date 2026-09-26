"""Selecting a subset of a campaign without changing its plan.

A six-hour four-GPU request does not schedule. Running the cells a paper
leads with first, in smaller jobs, gets results while the rest waits -- and
because the indices are the campaign's own, a later run fills in the rest
without recomputing anything.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "indices",
    Path(__file__).resolve().parent.parent
    / "scripts" / "list_phase_b_d3_task_indices.py",
)
indices = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(indices)

CONFIG = {
    "repetitions": [0, 1, 2],
    "cells": [
        {"benchmark_id": "phase_b_dalla_man_t1_canonical_named_easy", "tier": "easy"},
        {"benchmark_id": "phase_b_dalla_man_t1_canonical_named_hard", "tier": "hard"},
        {"benchmark_id": "phase_b_dalla_man_t2_canonical_named_easy", "tier": "easy"},
        {"benchmark_id": "phase_b_cstr_reactor_canonical_named_easy", "tier": "easy"},
    ],
}


def test_a_subset_keeps_the_campaign_indices() -> None:
    """The point: a later run completes the campaign without redoing this one."""
    everything = indices.task_indices(CONFIG, "all")
    subset = indices.task_indices(CONFIG, "all", "cstr")
    assert subset == (9, 10, 11)
    assert set(subset) <= set(everything)


def test_tier_and_name_filters_compose() -> None:
    assert indices.task_indices(CONFIG, "easy", "t1_") == (0, 1, 2)
    assert indices.task_indices(CONFIG, "hard", "t1_") == (3, 4, 5)
    # alternation selects several families at once
    assert indices.task_indices(CONFIG, "easy", "t1_|cstr") == (0, 1, 2, 9, 10, 11)


def test_no_match_selects_nothing_rather_than_everything() -> None:
    """A typo must not quietly launch the whole campaign."""
    assert indices.task_indices(CONFIG, "all", "nosuchcell") == ()


def test_an_absent_filter_is_unchanged_behaviour() -> None:
    assert indices.task_indices(CONFIG, "all") == tuple(range(12))
    assert indices.task_indices(CONFIG, "all", None) == tuple(range(12))


def test_the_real_config_yields_the_six_main_body_cells() -> None:
    """The cells the method campaign itself reports."""
    config = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "configs" / "phase_b_llm_ode_campaign_v1.json"
        ).read_text()
    )
    selected = indices.task_indices(config, "easy", "t1_|cstr|alien")
    assert len(selected) == 18
