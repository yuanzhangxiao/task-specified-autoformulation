#!/usr/bin/env python3
"""Audit saved v5 process replies; no LLM calls, fitting or promotion."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.process_assembly_audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.output)
    print(
        json.dumps({k: v for k, v in result.items() if k != "constructions"}, indent=2)
    )


if __name__ == "__main__":
    main()
