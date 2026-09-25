#!/usr/bin/env python3
"""Check saved public fits under current code; no fitting or campaign changes."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting.initialization_audit import audit_saved_initialization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, action="append", required=True)
    args = parser.parse_args()
    rows = []
    for directory in args.fit:
        try:
            rows.append({"status": "audited", **audit_saved_initialization(directory)})
        except (ValueError, KeyError, OSError) as error:
            rows.append(
                {
                    "status": "audit_error",
                    "directory": str(directory),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    print(
        json.dumps(
            {
                "protocol": "saved-initialization-namespace-audit-1",
                "rows": rows,
                "optimizer_calls": 0,
                "llm_calls": 0,
                "solver_rollouts": 0,
                "campaign_changes": 0,
                "test_data_opened": False,
            },
            indent=2,
            allow_nan=False,
        )
    )
    if any(row["status"] == "audit_error" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
