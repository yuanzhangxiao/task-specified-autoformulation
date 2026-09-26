"""Packaging evaluated models together with their metrics.

The pieces live in four files joined on two different keys. A bundle that got
the join wrong would pair one method's equations with another's score, which
is worse than having no bundle at all.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "package",
    Path(__file__).resolve().parent.parent / "scripts" / "package_phase_b_results.py",
)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def _evaluation(root: Path, *, method: str = "pysr") -> Path:
    """One evaluation root in the layout the pipeline writes."""
    (root / "frozen").mkdir(parents=True)
    (root / "adapted").mkdir()
    (root / "final-evaluation").mkdir()

    def lines(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
        )

    lines(root / "frozen" / "external_baseline_roster.jsonl", [
        {"request_id": "r0", "method_id": method, "benchmark_id": "cell_a",
         "tier": "easy", "repetition": 0, "artifact_status": "available",
         "source_path": "/some/result.json", "terminal_status": None,
         "reason": None},
        {"request_id": "r1", "method_id": method, "benchmark_id": "cell_a",
         "tier": "easy", "repetition": 1, "artifact_status": "missing",
         "source_path": "/gone.json", "terminal_status": "timed_out",
         "reason": "no model"},
    ])
    lines(root / "combined_source_outcomes.jsonl", [
        {"request_id": "r0", "status": "adapted", "subject_id": "s0"},
        {"request_id": "r1", "status": "missing", "subject_id": None},
    ])
    lines(root / "adapted" / "frozen_evaluation_subjects.jsonl", [
        {"subject_id": "s0", "candidate": {"state_equations": [
            {"state": "G", "rhs": "-0.5 * G + I"},
            {"state": "I", "rhs": "-0.2 * I"}]}},
    ])
    lines(root / "final-evaluation" / "final_evaluation_records.jsonl", [
        {"subject_id": "s0", "runtime": {"valid": True},
         "target_prediction": {"status": "available", "normalized_mse": 1.25},
         "public_mechanism": {"status": "available",
                              "evaluation": {"mechanism_compliance": 0.75,
                                             "mechanism_coverage": 1.0}},
         "complexity": {"state_count": 2, "latent_state_count": 0,
                        "parameter_count": 0, "additive_term_count": 3}},
    ])
    return root


def test_a_model_is_packaged_with_its_own_metrics(tmp_path: Path) -> None:
    rows, provenance = package.collect("sept", _evaluation(tmp_path / "eval"))

    assert len(rows) == 2
    scored = next(row for row in rows if row["request_id"] == "r0")
    assert scored["equations"] == {"G": "-0.5 * G + I", "I": "-0.2 * I"}
    assert scored["target_nmse"] == 1.25
    assert scored["mechanism_compliance"] == 0.75
    assert scored["complexity_terms"] == 3
    assert scored["runtime_valid"] is True
    assert scored["public_mechanism"]["evaluation"]["mechanism_compliance"] == 0.75
    assert scored["runtime_diagnostics"] == {"valid": True}
    assert provenance["planned"] == 2 and provenance["with_model"] == 1


def test_a_planned_identity_with_no_model_is_kept_not_dropped(
    tmp_path: Path,
) -> None:
    """Dropping it would shrink the denominator the medians depend on."""
    rows, _ = package.collect("sept", _evaluation(tmp_path / "eval"))

    missing = next(row for row in rows if row["request_id"] == "r1")
    assert missing["equations"] == {}
    assert missing["target_nmse"] is None
    assert missing["terminal_status"] == "timed_out"
    assert missing["adapter_status"] == "missing"


def test_every_source_file_is_identified_by_content(tmp_path: Path) -> None:
    """A transferred bundle has to be checkable against what produced it."""
    _, provenance = package.collect("sept", _evaluation(tmp_path / "eval"))
    assert "frozen/external_baseline_roster.jsonl" in provenance["files"]
    for digest in provenance["files"].values():
        assert len(digest) == 64


def test_metrics_are_absent_rather_than_zero_without_a_record() -> None:
    """A method that was never evaluated must not read as scoring zero."""
    blank = package.metrics_of(None)
    assert set(blank) == set(package.METRIC_FIELDS)
    assert all(value is None for value in blank.values())


def test_an_empty_evaluation_root_yields_nothing_rather_than_failing(
    tmp_path: Path,
) -> None:
    rows, provenance = package.collect("empty", tmp_path / "absent")
    assert rows == []
    assert provenance["planned"] == 0


def test_a_superseded_method_is_dropped_only_when_stated(tmp_path: Path) -> None:
    """An earlier run planned D3 before its campaign existed.

    All 120 of its identities were missing, and a later run produced the real
    ones. Merging both would count one method twice; choosing automatically by
    which rows carry models would silently pick a winner. So it is declared.
    """
    import subprocess
    import sys

    early = _evaluation(tmp_path / "early", method="d3_native_no_tools")
    late = _evaluation(tmp_path / "late", method="d3_native_no_tools")
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "package_phase_b_results.py"
    )

    def run(*extra: str):
        return subprocess.run(
            [sys.executable, str(script),
             "--evaluation", f"early={early}", "--evaluation", f"late={late}",
             "--out", str(tmp_path / "out"), *extra],
            capture_output=True, text=True, check=False,
        )

    # undeclared: refused, naming the method
    refused = run()
    assert refused.returncode != 0
    assert "appears under two evaluations" in refused.stderr

    # declared: the early rows are dropped and the bundle records that
    accepted = run("--supersede", "early=d3_native_no_tools")
    assert accepted.returncode == 0, accepted.stderr
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    early_entry = next(
        item for item in manifest["evaluations"] if item["label"] == "early"
    )
    assert early_entry["superseded_methods"] == ["d3_native_no_tools"]
    assert early_entry["rows_superseded"] == 2
    assert manifest["planned_total"] == 2
    for name, digest in manifest["payload_sha256"].items():
        assert package.sha256(tmp_path / "out" / name) == digest


def _subject_with_parameters(root: Path) -> None:
    """Rewrite the fixture's subject as a D3-shaped one: fitted, discrete."""
    (root / "adapted" / "frozen_evaluation_subjects.jsonl").write_text(
        json.dumps(
            {
                "subject_id": "s0",
                "execution_semantics": "discrete_increment_recursive_rollout",
                "candidate": {
                    "candidate_id": "d3_abc",
                    "state_equations": [{"state": "G", "rhs": "k_g * G + k_m * u"}],
                    "observation_mappings": [{"channel": "G", "expression": "G"}],
                    "initial_conditions": [{"state": "G", "scope": "global",
                                            "expression": "G"}],
                },
                "parameterization": {
                    "status": "available",
                    "global_parameters": {"k_g": -0.5, "k_m": 0.2},
                    "global_initial_conditions": {"G": 90.0},
                },
                "validation_context": {"targets": ["G"], "auxiliaries": ["u"]},
                "source_provenance": {"path": "/some/result.json"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_the_package_carries_what_replaying_a_model_requires(
    tmp_path: Path,
) -> None:
    """Equations alone are an incomplete model for a method that fits apart.

    D3's coefficients live in the parameterization, not in its right-hand
    sides, and its execution semantics are discrete: replaying it through an
    ODE solver would reinterpret what it learned.
    """
    root = _evaluation(tmp_path / "eval", method="d3_native_no_tools")
    _subject_with_parameters(root)
    rows, _ = package.collect("d3", root)

    scored = next(row for row in rows if row["request_id"] == "r0")
    model = scored["model"]
    assert model["parameterization"]["global_parameters"] == {"k_g": -0.5, "k_m": 0.2}
    assert model["parameterization"]["global_initial_conditions"] == {"G": 90.0}
    assert model["execution_semantics"] == "discrete_increment_recursive_rollout"
    assert model["validation_context"]["auxiliaries"] == ["u"]
    # the candidate is whole, not reduced to its right-hand sides
    assert model["candidate"]["observation_mappings"]
    assert model["candidate"]["initial_conditions"]
    # and the flat summary says how much was fitted, without the nesting
    assert scored["fitted_parameter_count"] == 2
    assert scored["fitted_initial_condition_count"] == 1
    assert scored["parameterization_status"] == "available"


def test_an_identity_without_a_model_carries_no_replay_payload(
    tmp_path: Path,
) -> None:
    rows, _ = package.collect("sept", _evaluation(tmp_path / "eval"))
    missing = next(row for row in rows if row["request_id"] == "r1")
    assert missing["model"] is None
    assert missing["fitted_parameter_count"] == 0
    assert missing["execution_semantics"] is None


def test_per_requirement_evidence_and_runtime_errors_are_not_flattened_away(
    tmp_path: Path,
) -> None:
    root = _evaluation(tmp_path / "eval")
    path = root / "final-evaluation" / "final_evaluation_records.jsonl"
    record = json.loads(path.read_text())
    record["public_mechanism"]["evaluation"]["graph_mechanism_results"] = [
        {"mechanism_id": "balance", "status": "ambiguous",
         "predicates": [{"predicate": "graph_inference", "status": "ambiguous"}]}
    ]
    record["runtime"] = {"valid": False, "errors": ["unused declaration"]}
    path.write_text(json.dumps(record) + "\n")
    rows, _ = package.collect("sept", root)
    assert rows[0]["public_mechanism"] == record["public_mechanism"]
    assert rows[0]["runtime_diagnostics"] == record["runtime"]
    assert rows[1]["public_mechanism"] is None
    assert rows[1]["runtime_diagnostics"] is None
