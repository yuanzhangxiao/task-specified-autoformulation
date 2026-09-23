#!/usr/bin/env python3
"""Offline saved-request import, paired mocked rejudge, and deterministic resume."""

import json
import tempfile
from pathlib import Path

from tests.test_judge_sign_recheck import smoke

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="judge-sign-smoke-") as directory:
        print(json.dumps(smoke(Path(directory)), indent=2))
