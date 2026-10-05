#!/usr/bin/env python3
"""Audit saved public topology drafts without LLM calls, fitting or model changes."""

import argparse
import json
from pathlib import Path

from autoformalism.research.construction_graph_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"inputs", "audit_runtime", "rows", "public_contracts"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
