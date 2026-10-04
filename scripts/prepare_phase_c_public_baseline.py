#!/usr/bin/env python3
"""Freeze the Phase C classical baseline matrix against one verified release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.phase_c_baseline_plan import freeze_phase_c_baseline_plan


def main() -> None:
    """Verify the release receipt and prompts, then write the task ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_phase_c_baseline_plan(
        args.config, args.output_root, args.release
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
