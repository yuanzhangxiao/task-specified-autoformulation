#!/usr/bin/env python3
"""Reaggregate saved mechanism outcomes as case means and sample SD; no probes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write


def moments(values: list[float]) -> dict:
    """Describe the given units; SD is undefined for fewer than two units."""
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else None,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    """Equal weight per planned case, averaging requirements then repetitions."""
    seen = set()
    for row in rows:
        key = row["method"], row["benchmark_id"], row["repetition"]
        if key in seen:
            raise ValueError(f"duplicate planned repetition: {key}")
        seen.add(key)
        outcomes = [m["status"] for m in row["mechanisms"]]
        if not outcomes or set(outcomes) - {"pass", "fail", "unresolved"}:
            raise ValueError("require nonempty pass/fail/unresolved outcomes")
    methods = []
    for method in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == method]
        cases = []
        for case in sorted({r["benchmark_id"] for r in selected}):
            runs = [r for r in selected if r["benchmark_id"] == case]
            lower, upper = [], []
            counts = Counter()
            for row in runs:
                c = Counter(m["status"] for m in row["mechanisms"])
                counts.update(c)
                lower.append(c["pass"] / c.total())
                upper.append((c["pass"] + c["unresolved"]) / c.total())
            cases.append(
                {
                    "benchmark_id": case,
                    "runs": len(runs),
                    "counts": dict(counts),
                    "confirmed": moments(lower),
                    "possible": moments(upper),
                }
            )
        methods.append(
            {
                "method": method,
                "cases": cases,
                "planned_runs": len(selected),
                "benchmark_count": len(cases),
                "counts": dict(
                    Counter(m["status"] for r in selected for m in r["mechanisms"])
                ),
                "confirmed": moments([c["confirmed"]["mean"] for c in cases]),
                "possible": moments([c["possible"]["mean"] for c in cases]),
            }
        )
    return methods


def display(value: dict) -> str:
    """Percent mean and percentage-point sample SD."""
    if value["mean"] is None:
        return "unavailable"
    sd = f"{100 * value['sd']:.1f}" if value["sd"] is not None else "unavailable"
    return f"{100 * value['mean']:.1f} ({sd})"


def markdown(methods: list[dict]) -> str:
    """Keep case coverage and unresolved counts next to headline scores."""
    lines = [
        "# Public-mechanism compliance: mean (SD)",
        "",
        "Mean over repetitions, then equal-weight mean over planned benchmarks.",
        "Headline SD is sample SD across benchmark means; per-case SD is across runs.",
        "Possible scores are evidence bounds, not confidence intervals.",
        "Different benchmark counts are different scopes and must not be pooled.",
        "",
        "| Method | Cases | Confirmed % (SD pp) | Possible % (SD pp) | P/F/U |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in methods:
        c = item["counts"]
        lines.append(
            f"| {item['method']} | {item['benchmark_count']} | "
            f"{display(item['confirmed'])} | {display(item['possible'])} | "
            f"{c.get('pass', 0)}/{c.get('fail', 0)}/{c.get('unresolved', 0)} |"
        )
    return "\n".join(lines) + "\n"


def report(root: Path, output: Path) -> dict:
    """Verify saved seals without requiring the old numerical runtime to run again."""
    if output.resolve() == root.resolve():
        raise ValueError("use a separate reporting directory")
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != "fitted-public-mechanism-tests-1":
        raise ValueError("unsupported assessment protocol")
    rows, hashes = [], {}
    for source in plan["rows"]:
        path = root / "results" / f"{source['index']:04d}" / "result.json"
        saved = sealed_read(path)  # Missing workers cannot disappear from the roster.
        if (saved["plan_sha256"], saved["index"]) != (
            plan["artifact_sha256"],
            source["index"],
        ):
            raise ValueError("result/plan mismatch")
        expected = [r["id"] for r in source["mechanisms"]]
        actual = [r["id"] for r in saved["mechanisms"]]
        if sorted(actual) != sorted(expected):
            raise ValueError("result mechanism roster differs")
        hashes[str(source["index"])] = saved["artifact_sha256"]
        rows.append(
            {
                **{k: source[k] for k in ("method", "benchmark_id", "repetition")},
                "assessment_status": saved["status"],
                "mechanisms": saved["mechanisms"],
            }
        )
    value = {
        "protocol": "mechanism-case-mean-sample-sd-1",
        "source_plan_sha256": plan["artifact_sha256"],
        "source_result_sha256": hashes,
        "methods": aggregate(rows),
        "rows": rows,
        "status_counts": dict(Counter(r["assessment_status"] for r in rows)),
        "sd_scope": "sample SD across benchmark means, ddof=1",
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    result = sealed_write(output / "summary.json", value)
    (output / "SUMMARY.md").write_text(markdown(result["methods"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root, args.output)
    print(
        json.dumps(
            {"status_counts": result["status_counts"], "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
