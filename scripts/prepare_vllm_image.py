#!/usr/bin/env python3
"""Prepare the pinned-version vLLM SIF on a CPU node; no model calls or fitting."""

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.vllm_image_bootstrap import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.image, args.scratch), indent=2))
