"""The statistic Table 1 specifies, which is not the roster report's.

EXPERIMENTS_DRAFT_NOTES.md asks for a macro median over nine equally weighted
cases, with case medians taken over repetitions first, and an unscaled MAD.
The roster report's median is over every row across forty cells. Transcribing
one into the other would be a different number under the same name.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "table",
    Path(__file__).resolve().parent.parent
    / "scripts" / "build_experiments_table_results.py",
)
table = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(table)


def _rows(method: str, values: dict[str, list[float | None]]) -> list[dict]:
    rows = []
    for case, samples in values.items():
        for repetition, value in enumerate(samples):
            rows.append({
                "method": method, "benchmark_id": case, "tier": "easy",
                "repetition": repetition, "target_nmse": value,
                "complexity_terms": 4 if value is not None else None,
                "mechanism_compliance": 0.5 if value is not None else None,
            })
    return rows


def test_case_medians_come_before_the_macro_summary() -> None:
    """Pooling repetitions across cases would weight cases by their coverage."""
    assert table.case_median([1.0, 3.0, 5.0]) == 3.0
    assert table.case_median([]) is None
    centre, spread = table.macro([1.0, 3.0, 5.0, 7.0, 9.0])
    assert centre == 5.0
    # unscaled MAD: median of |1-5|,|3-5|,|5-5|,|7-5|,|9-5| = median(4,2,0,2,4)
    assert spread == 2.0


def test_a_method_missing_a_whole_case_has_no_headline_aggregate() -> None:
    """Computing one over the cases it covered compares different rosters.

    SINDy has no evaluable run for the named CSTR case, which is one of the
    nine, so this is the real situation and not a hypothetical.
    """
    assert table.macro([1.0, 2.0, None]) == (None, None)
    assert table.render(None, None) == r"\textbf{TBD}"


def test_the_rendered_value_carries_its_spread() -> None:
    assert table.render(2.4655, 0.5) == "2.466 [0.500]"
    assert table.render(6.0, 1.0, digits=1) == "6.0 [1.0]"


def test_coverage_counts_only_the_nine_cases(tmp_path: Path) -> None:
    """A method's coverage must not be inflated by cells outside the roster."""
    rows = _rows("pysr", {table.NINE_CASES[0]: [1.0, 2.0, None]})
    rows += _rows("pysr", {"phase_b_dalla_man_t3_canonical_named_easy": [1.0, 1.0]})
    assert table.coverage(rows, "pysr") == (2, 3)


def test_every_named_case_appears_in_the_real_campaign_config() -> None:
    """A typo in the roster would silently drop a case from the table."""
    config = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "configs" / "phase_b_llm_ode_campaign_v1.json"
        ).read_text()
    )
    known = {str(cell["benchmark_id"]) for cell in config["cells"]}
    assert len(table.NINE_CASES) == 9
    assert set(table.NINE_CASES) <= known
