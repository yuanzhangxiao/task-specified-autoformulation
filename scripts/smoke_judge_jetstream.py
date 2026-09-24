#!/usr/bin/env python3
"""Offline real-schema paired judge, mocked HTTP, accounting and resume smoke."""

import json
import tempfile
from pathlib import Path

from tests.test_judge_jetstream import smoke

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="jetstream-judge-smoke-") as directory:
        print(json.dumps(smoke(Path(directory)), indent=2))
