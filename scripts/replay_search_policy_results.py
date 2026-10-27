#!/usr/bin/env python3
"""Recheck pinned search-policy replies without replaying a new search trajectory.

The original Downloads archives are no longer present. Complete preserved member
trees have their own literal pins; original archive hashes are provenance only.
No archived code, provider, fitter, rollout or held-out table is executed/opened.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import stat
from collections import Counter
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from autoformalism.llm.staged_topology import atomic_json, visible_response
from autoformalism.rebuttal.prefit_construction_campaign import _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import construction_search_policy as engine
from autoformalism.research.construction_affine_family_consistency import (
    COMMON_ZERO_FORCING_POLICY,
)
from autoformalism.research.interaction_source import _Reader
from autoformalism.staged_topology import content_hash
from scripts import structural_revision_campaign as base

PROTOCOL = "saved-search-policy-results-recheck-2"
MAX_FILES = 4096
MAX_BYTES = 128 * 1024 * 1024
SOURCE_PINS = {
    "20b": {
        "preserved_tree_sha256": (
            "b0db71972afb552670b133ee8ef07da1651819d328687c4e2226f206d8b66292"
        ),
        "plan_sha256": (
            "f5324732908b7714e0daaef1286a62194d577bdbcbf5e9ffe79d7a435efe6743"
        ),
        "original_archive_sha256": (
            "a3ab50efe5c22460f5cb63d45e46f1a9584330aede05114731a52278e39a86a7"
        ),
    },
    "120b": {
        "preserved_tree_sha256": (
            "66a393281fa731754b8026623e9c9b2a263c621aff5a68ef2e4e02babc449bea"
        ),
        "plan_sha256": (
            "6583ef7f25f87f0cb2ccbd375ba2578dec2263caafd9fe0c9709ef9dc406e156"
        ),
        "original_archive_sha256": (
            "44c667ed2ef10ec75787e40e4758eb9c98fb71f5daa83e344b5ad29649f04be3"
        ),
    },
}
SMOKE_SOURCE_PINS = {
    "20b": {
        "preserved_tree_sha256": (
            "6e91e8ed6ce1f1900314f83bdabb9900feec95c6fd9db082064247402f53f145"
        ),
        "plan_sha256": (
            "3e7327a9e32f8315732fe57b2bebc4e6c1db94ffee5046f442d17ec7fec6b238"
        ),
        "original_archive_sha256": (
            "11e4149c03ce8f2fccd29e05c1ec5898a07ac880f5328adcc21d91ade6e03de2"
        ),
    },
    "120b": {
        "preserved_tree_sha256": (
            "b736df5894deb62dd2df6708216fe474a9a9835b13f11952ca009758686da649"
        ),
        "plan_sha256": (
            "925501ba2f6e4331bfef3c4fe2033fb1a38345b2fa2b1b6c0c2e70f1765b9232"
        ),
        "original_archive_sha256": (
            "7cb90817342371b9613d941aac5f95236c2a7f6e523f6d0d6b2d17104791f67b"
        ),
    },
}


BREADTH_SMOKE_SOURCE_PINS = {
    "20b": {
        "preserved_tree_sha256": (
            "b46191a52fd210778a19069937b55371dea07b78afbf730a096af1bbbe136788"
        ),
        "plan_sha256": (
            "d9b3515c343e7ee843dfabaf4b484b28e7b235dca027787d6a8ad0e19f06affb"
        ),
        "original_archive_sha256": (
            "7d039fc7b81a3c9901534c5eb28b3767b9a014bb0e5fb24150410cf6748fc1fb"
        ),
    },
    "120b": {
        "preserved_tree_sha256": (
            "368078268aab43e7aac2b56b14d86b67101a4e051217a29f8f85648068fad5df"
        ),
        "plan_sha256": (
            "d208d65cc3553c7013ece61a08dbe45bf0b806121001e8329f2a31e1f2451ae8"
        ),
        "original_archive_sha256": (
            "599667a5e8880beb622c6a55f6f9ca215a1228779248260cc2eb2d2e97ab50be"
        ),
    },
}
SOURCE_PROTOCOLS = {
    "v1": "construction-search-policy-1",
    "v2-smoke": "construction-search-policy-2",
    "v3-smoke": "construction-search-policy-3",
}


def _source_pins(version: str) -> dict:
    if version == "v1":
        return SOURCE_PINS
    if version == "v2-smoke":
        return SMOKE_SOURCE_PINS
    if version == "v3-smoke":
        return BREADTH_SMOKE_SOURCE_PINS
    raise ValueError("unknown pinned source version")


def tree_manifest(root: Path) -> dict[str, str]:
    """Hash a bounded regular tree, rejecting links before reading file content."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("a preserved regular source directory is required")
    files, total, entries = {}, 0, 0
    pending = [root]
    while pending:
        for path in sorted(pending.pop().iterdir()):
            entries += 1
            mode = path.lstat().st_mode
            if entries > MAX_FILES * 2:
                raise ValueError("source directory exceeds entry bound")
            if stat.S_ISDIR(mode):
                pending.append(path)
            elif stat.S_ISREG(mode):
                total += path.stat().st_size
                if len(files) >= MAX_FILES or total > MAX_BYTES:
                    raise ValueError("preserved source exceeds resource bounds")
                files[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            else:
                raise ValueError("source tree contains a link or special file")
    return dict(sorted(files.items()))


def _historical_patches(record: dict, attempt: dict, version: str = "v1") -> list[dict]:
    """Read the historical envelope, independently of today's exactly-three rule."""
    if attempt.get("parse_error") is not None:
        raise ValueError("this pinned replay requires readable historical replies")
    raw = visible_response(record)
    if attempt["mode"] == "options":
        if not isinstance(raw, dict) or not isinstance(raw.get("options"), list):
            raise ValueError("historical option envelope differs")
        options = raw["options"]
        if (version == "v3-smoke" and len(options) != 5) or (
            version != "v3-smoke" and not 1 <= len(options) <= 3
        ):
            raise ValueError("historical option count differs")
        return [option["patch"] for option in options]
    return [raw]


def load_source(root: Path, arm: str, version: str = "v1") -> dict:
    """Authenticate saved parents, request bodies, tokenizations and completions."""
    manifest = tree_manifest(root)
    pins = _source_pins(version)[arm]
    if content_hash(manifest) != pins["preserved_tree_sha256"]:
        raise ValueError("preserved source tree identity differs")
    reader = _Reader(root)
    plan = reader.json("plan.json", "artifact_sha256")
    selected = reader.json("selected_sources.json", "artifact_sha256")
    if (
        plan["protocol"] != SOURCE_PROTOCOLS[version]
        or plan["artifact_sha256"] != pins["plan_sha256"]
        or plan["model_size"] != arm
        or plan["selected_sources_sha256"] != selected["artifact_sha256"]
        or hashlib.sha256(reader.read("saved-source.tar.gz")).hexdigest()
        != plan["source_archive_sha256"]
        or plan["test_data_opened"] is not False
    ):
        raise ValueError("historical plan or selected-source identity differs")
    base.old._validate_inputs(selected["inputs"])
    candidates, records_all, option_counts = [], [], []
    seen_cases = set()
    for entry in plan["cases"]:
        case, name = entry["case_id"], entry["parent_id"]
        if case in seen_cases:
            raise ValueError("duplicate historical case")
        seen_cases.add(case)
        parent = selected["parents"][name]
        if entry["input_sha256"] != content_hash(parent):
            raise ValueError("historical parent identity differs")
        control = reader.json(f"controls/{name}.json", "artifact_sha256")
        if (
            control["problem"] != parent["problem"]
            or control["problem_sha256"] != content_hash(parent["problem"])
            or control["control_unchanged"] is not True
        ):
            raise ValueError("historical parent control differs")
        prefix = f"results/{case}"
        if (
            version in {"v2-smoke", "v3-smoke"}
            and not (root / prefix / "completion.json").exists()
        ):
            if (root / prefix).exists():
                raise ValueError("unsealed partial episode in pinned smoke source")
            continue
        result = reader.json(f"{prefix}/result.json")
        receipt = reader.json(f"{prefix}/completion.json", "artifact_sha256")
        records = base.sources._records(reader, entry, plan)
        tokens = base.sources._records(reader, entry, plan, tokens=True)
        if (
            receipt["plan_sha256"] != plan["artifact_sha256"]
            or receipt["input_sha256"] != entry["input_sha256"]
            or receipt["calls_sha256"] != content_hash(records)
            or receipt["tokenizations_sha256"] != content_hash(tokens)
            or receipt["status"] != result["status"]
            or "result.json" not in receipt["files"]
        ):
            raise ValueError("historical completion receipt differs")
        for filename, digest in receipt["files"].items():
            if "/" in filename or not (
                filename == "result.json"
                or (filename.startswith("fit-call") and filename.endswith(".json"))
            ):
                raise ValueError("unexpected historical completion file")
            if (
                hashlib.sha256(reader.read(f"{prefix}/{filename}")).hexdigest()
                != digest
            ):
                raise ValueError("historical completed file differs")
        attempts = result["attempts"]
        if (
            [a["attempt"] for a in attempts] != list(range(len(attempts)))
            or len(attempts) != len(records)
            or result["generation_calls"] != len(records)
            or result["candidate_count"] != len(result["candidates"])
            or {a["request_hash"] for a in attempts}
            != {r["request_hash"] for r in records}
        ):
            raise ValueError("historical call roster differs")
        emitted = []
        by_id = {candidate["id"]: candidate for candidate in result["candidates"]}
        if len(by_id) != len(result["candidates"]):
            raise ValueError("duplicate historical candidate identity")
        for attempt in attempts:
            record = next(
                r for r in records if r["request_hash"] == attempt["request_hash"]
            )
            payload = json.loads(record["request"]["body"]["messages"][1]["content"])
            if (
                record["attempt"] != attempt["attempt"]
                or record["step"] != "search_policy"
                or payload["current_model"]["problem_sha256"]
                != content_hash(parent["problem"])
                or payload["public_task"] != parent["public_brief"]
            ):
                raise ValueError("historical request context differs")
            patches = _historical_patches(record, attempt, version)
            expected_ids = [
                f"call{attempt['attempt']}-option{i}" for i in range(len(patches))
            ]
            if attempt["candidate_ids"] != expected_ids:
                raise ValueError("historical candidate roster differs")
            if attempt["mode"] == "options":
                option_counts.append({"case_id": case, "count": len(patches)})
            for index, raw in enumerate(patches):
                identifier = expected_ids[index]
                old = by_id[identifier]
                if (
                    old["raw_reply"] != raw
                    or old["call_index"] != attempt["attempt"]
                    or old["option_index"] != index
                ):
                    raise ValueError("historical raw patch differs from provider reply")
                emitted.append(identifier)
                candidates.append(
                    {
                        "case_id": case,
                        "parent_id": name,
                        "seed": entry["seed"],
                        "policy": entry["policy"],
                        "candidate_id": identifier,
                        "request_sha256": record["request_hash"],
                        "record_sha256": content_hash(record),
                        "historical": old,
                    }
                )
        if emitted != [candidate["id"] for candidate in result["candidates"]]:
            raise ValueError("historical candidate order differs")
        records_all.extend(records)
    if version in {"v2-smoke", "v3-smoke"}:
        completed = {c["case_id"] for c in candidates}
        first_parent = min(row["parent_id"] for row in plan["cases"])
        expected = {
            row["case_id"]
            for row in plan["cases"]
            if row["parent_id"] == first_parent and row["seed"] == 0
        }
        if completed != expected or len(expected) != (
            1 if version == "v3-smoke" else 3
        ):
            raise ValueError("pinned smoke completion roster differs")
    return {
        "arm": arm,
        "plan": plan,
        "selected": selected,
        "candidates": candidates,
        "option_counts": option_counts,
        "historical_cost": _cost(records_all),
        "source_manifest": manifest,
        "source_version": version,
    }


def _outcome(candidate: dict) -> str:
    return (candidate.get("diagnostic") or {}).get("code", candidate["status"])


def _render(root: Path, summary: dict, rows: list[dict], parents: dict) -> None:
    """Render every audited patch and prospective feedback without an LLM call."""
    style = (
        "body{max-width:1180px;margin:2rem auto;font:16px/1.5 sans-serif}"
        "pre{white-space:pre-wrap;background:#f5f6f8;padding:1em;"
        "overflow-wrap:anywhere}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccd;padding:.45rem}"
    )

    def pre(value):
        return (
            "<pre>"
            + html.escape(json.dumps(value, ensure_ascii=False, indent=2))
            + "</pre>"
        )

    parts = [
        f"<!doctype html><meta charset='utf-8'><style>{style}</style>",
        "<h1>Offline search reply recheck</h1><p>"
        + html.escape(summary["interpretation"])
        + "</p>",
        "<p><a href='PROMPTS.html'>Updated prompt examples</a></p>",
        "<h2>Totals and original costs</h2>" + pre(summary["arms"]),
        "<h2>Every saved candidate</h2><table><tr><th>Model / case / candidate</th>"
        "<th>Historical</th><th>Rechecked</th><th>Detail</th></tr>",
    ]
    prompt_parts = [
        f"<!doctype html><meta charset='utf-8'><style>{style}</style>",
        "<h1>Updated search prompts: offline preview</h1>",
        "<p>These requests were not sent. Historical replies did not see "
        "this wording. These examples retain the historical full parent-relative "
        "patch reply contract, not the prospective branch-delta contract.</p>",
        "<h2>Changes from the previous prompt</h2><ul>"
        "<li><mark>Exactly five alternatives are mandatory in the new breadth "
        "pilot.</mark> A reason "
        "for fewer options no longer satisfies the batch contract.</li>"
        "<li><mark>A complete intended patch is a delta against the parent, "
        "not a complete restatement of its model.</mark> Untouched components "
        "remain in place; rejected edits are not silently installed.</li>"
        "<li><mark>Unambiguous identical declarations can be normalized.</mark> "
        "Type, role, scope, domain or definition conflicts still need repair.</li>"
        "<li><mark>Shared symbolic initial maps and observed initial readouts "
        "are combined when qualified.</mark> A trajectory initial may be "
        "constrained by both; shared unknowns are not freely fitted per row."
        "</li></ul>",
        "<h2>System</h2>" + pre(engine.SYSTEM),
        "<h2>Exactly-five response schema</h2>"
        + pre(engine.options_reply_model(5).model_json_schema()),
    ]
    seen_diagnostics = set()
    for row in rows:
        old, new = row["historical"], row["candidate"]
        key = f"{row['arm']}-{row['case_id']}-{row['candidate_id']}"
        parent = parents[(row["arm"], row["parent_id"])]
        feedback = (
            json.loads(
                engine.request(
                    parent,
                    row["policy"],
                    history=[new],
                    intended=new,
                    options_count=5,
                    maximum_calls=5,
                    common_zero_forcing=summary["common_zero_forcing"],
                )
            )
            if new["status"] == "rejected"
            else None
        )
        body = [
            f"<!doctype html><meta charset='utf-8'><style>{style}</style>",
            "<p><a href='../REPORT.html'>All candidates</a></p><h1>"
            + html.escape(key)
            + "</h1>",
            "<p>Historical: "
            + html.escape(_outcome(old))
            + "; rechecked: "
            + html.escape(_outcome(new))
            + ". No adoption or fitting.</p>",
            "<h2>Original parent</h2>"
            + pre(
                engine._model_display(
                    parent["problem"], parent.get("scientific_aliases")
                )
            ),
            "<h2>Original raw patch</h2>" + pre(old["raw_reply"]),
        ]
        if new.get("problem"):
            body += [
                "<h2>Rechecked trial equations and initialization</h2>"
                + pre(
                    engine._model_display(
                        new["problem"], new.get("audit", {}).get("scientific_aliases")
                    )
                )
            ]
        body += [
            "<h2>Current checks and normalization</h2>"
            + pre({"diagnostic": new.get("diagnostic"), "audit": new.get("audit")})
        ]
        if feedback is not None:
            body += ["<h2>Prospective repair request</h2>" + pre(feedback)]
            diagnostic = _outcome(new)
            if diagnostic not in seen_diagnostics:
                seen_diagnostics.add(diagnostic)
                prompt_parts += [
                    "<h2>"
                    + html.escape(diagnostic)
                    + "</h2><p>Example: "
                    + html.escape(key)
                    + "</p>"
                    + pre(feedback)
                ]
        (root / "rows" / f"{key}.html").write_text("\n".join(body))
        parts += [
            "<tr><td>"
            + html.escape(key)
            + "</td><td>"
            + html.escape(_outcome(old))
            + "</td><td>"
            + html.escape(_outcome(new))
            + f"</td><td><a href='rows/{key}.html'>"
            "Equations, patch, checks and feedback</a></td></tr>"
        ]
    if parents:
        first = next(iter(parents.values()))
        prompt_parts += [
            "<h2>Initial five-option request</h2>"
            + pre(
                json.loads(
                    engine.request(
                        first,
                        "options_shallow",
                        batch=True,
                        options_count=5,
                        maximum_calls=5,
                        common_zero_forcing=summary["common_zero_forcing"],
                    )
                )
            )
        ]
    parts += ["</table>"]
    (root / "REPORT.html").write_text("\n".join(parts))
    (root / "PROMPTS.html").write_text("\n".join(prompt_parts))


def replay(
    sources: dict[str, Path],
    root: Path,
    version: str = "v1",
    *,
    common_zero_forcing: bool = False,
) -> dict:
    """Checkpoint independent gate rechecks; never advance historical decisions."""
    if type(common_zero_forcing) is not bool:
        raise ValueError("common_zero_forcing must be boolean")
    root = root.resolve()
    if any(
        root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(root)
        for p in sources.values()
    ):
        raise ValueError("recheck output and preserved sources must be separate")
    loaded = [load_source(sources[arm], arm, version) for arm in sorted(sources)]
    if len({source["plan"]["comparison_sha256"] for source in loaded}) != 1:
        raise ValueError("historical comparisons differ")
    identity = {
        "protocol": PROTOCOL,
        "engine_protocol": engine.PROTOCOL,
        "source_version": version,
        "common_zero_forcing": common_zero_forcing,
        "common_zero_forcing_policy": (
            COMMON_ZERO_FORCING_POLICY if common_zero_forcing else None
        ),
        "reply_interpretation": "historical full parent-relative patch; "
        "no branch-delta reinterpretation",
        "sources": {
            source["arm"]: _source_pins(version)[source["arm"]] for source in loaded
        },
        "runtime_source_sha256": runtime_source_hash(),
        "entrypoints": base._entrypoints(),
        "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "llm_calls": 0,
        "optimizer_calls": 0,
        "solver_rollouts": 0,
        "test_data_opened": False,
        "adopted_models": 0,
        "fitting_exports": 0,
    }
    with _lock(root / "recheck-lock"):
        plan = sealed_write(root / "plan.json", identity)
        rows, parents = [], {}
        for source in loaded:
            arm = source["arm"]
            sealed_write(
                root / f"source-manifest-{arm}.json", source["source_manifest"]
            )
            for saved in source["candidates"]:
                parent = base._parent(source["selected"], saved["parent_id"])
                parents[(arm, saved["parent_id"])] = parent
                receipt = {
                    "plan_sha256": plan["artifact_sha256"],
                    "arm": arm,
                    "historical_candidate_sha256": content_hash(saved["historical"]),
                    **{k: v for k, v in saved.items() if k != "historical"},
                }
                path = (
                    root
                    / "rows"
                    / f"{arm}-{saved['case_id']}-{saved['candidate_id']}.json"
                )
                if path.exists():
                    row = sealed_read(path)
                    if row["receipt"] != receipt:
                        raise ValueError("saved recheck input identity differs")
                else:
                    old = saved["historical"]
                    candidate = engine.assess_candidate(
                        parent,
                        old["raw_reply"],
                        candidate_id=old["id"],
                        call_index=old["call_index"],
                        option_index=old["option_index"],
                        branch_id=old["branch_id"],
                        premise=old["premise_changed"],
                        affected_target=old["affected_target"],
                        common_zero_forcing=common_zero_forcing,
                    )
                    row = sealed_write(
                        path,
                        {
                            **receipt,
                            "receipt": receipt,
                            "historical": old,
                            "candidate": candidate,
                            "adopted": False,
                            "fitting_exported": False,
                        },
                    )
                rows.append(row)
        arms = {}
        for source in loaded:
            arm_rows = [r for r in rows if r["arm"] == source["arm"]]
            arms[source["arm"]] = {
                "candidates": len(arm_rows),
                "historical_cost": source["historical_cost"],
                "historical_outcomes": dict(
                    Counter(_outcome(r["historical"]) for r in arm_rows)
                ),
                "rechecked_outcomes": dict(
                    Counter(_outcome(r["candidate"]) for r in arm_rows)
                ),
                "changed_outcomes": sum(
                    _outcome(r["historical"]) != _outcome(r["candidate"])
                    for r in arm_rows
                ),
                "eligible_before": sum(
                    r["historical"]["status"] == "eligible" for r in arm_rows
                ),
                "eligible_after": sum(
                    r["candidate"]["status"] == "eligible" for r in arm_rows
                ),
                "newly_eligible": sum(
                    r["historical"]["status"] != "eligible"
                    and r["candidate"]["status"] == "eligible"
                    for r in arm_rows
                ),
                "newly_blocked": sum(
                    r["historical"]["status"] == "eligible"
                    and r["candidate"]["status"] != "eligible"
                    for r in arm_rows
                ),
                "historical_option_counts": dict(
                    Counter(str(r["count"]) for r in source["option_counts"])
                ),
                "historical_batches_short_of_new_rule": sum(
                    r["count"] != (5 if version == "v3-smoke" else 3)
                    for r in source["option_counts"]
                ),
            }
        summary = {
            **identity,
            "plan_sha256": plan["artifact_sha256"],
            "arms": arms,
            "interpretation": (
                "Independent saved-patch rechecks, not live policy results. Later "
                "historical replies did not see the new feedback. V1 option "
                "batches could contain fewer than three; v2 smoke required three; "
                "v3 smoke required five. Raw replies retain their original "
                "parent-relative patch semantics; branch-delta composition is a "
                "separate counterfactual. "
                "Historical counts are reported separately, not retroactive "
                "failures under the five-option pilot. Eligibility retains "
                "deferred checks and does "
                "not establish scientific validity or fitting accuracy. "
                "No models adopted or exported for fitting."
            ),
            "rows": [
                {
                    "arm": r["arm"],
                    "case_id": r["case_id"],
                    "candidate_id": r["candidate_id"],
                    "historical": _outcome(r["historical"]),
                    "rechecked": _outcome(r["candidate"]),
                }
                for r in rows
            ],
        }
        atomic_json(root / "summary.json", summary)
        _render(root, summary, rows, parents)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-20b", type=Path, required=True)
    parser.add_argument("--source-120b", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-version", choices=tuple(SOURCE_PROTOCOLS), default="v1"
    )
    parser.add_argument(
        "--common-zero-forcing",
        action="store_true",
        help="Enable the prospective qualified common-zero-input proof; "
        "historical defaults stay unchanged.",
    )
    args = parser.parse_args()
    result = replay(
        {"20b": args.source_20b, "120b": args.source_120b},
        args.root,
        args.source_version,
        common_zero_forcing=args.common_zero_forcing,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
