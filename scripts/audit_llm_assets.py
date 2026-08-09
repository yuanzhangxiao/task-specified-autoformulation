"""Create a deterministic manifest of reusable LLM cache entries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from autoformalism.rebuttal.llm_assets import audit_llm_caches, resolve_llm_caches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resolved-cache-root", type=Path)
    parser.add_argument(
        "--fail-on-conflict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    audit = audit_llm_caches(tuple(args.roots))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = audit.model_dump(mode="json", exclude={"records"})
    (args.output_root / "llm_cache_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = (
        tuple(type(audit.records[0]).model_fields)
        if audit.records
        else (
            "path",
            "request_hash",
            "provider",
            "model",
            "parsed_digest",
            "raw_digest",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
    )
    with (args.output_root / "llm_cache_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in audit.records:
            writer.writerow(record.model_dump(mode="json"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.resolved_cache_root is not None:
        resolution = resolve_llm_caches(audit, args.resolved_cache_root)
        resolution_payload = resolution.model_dump(mode="json")
        (args.output_root / "llm_cache_resolution.json").write_text(
            json.dumps(resolution_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(resolution_payload, indent=2, sort_keys=True))
    if args.fail_on_conflict and (
        audit.semantic_conflict_count or audit.malformed_files
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
