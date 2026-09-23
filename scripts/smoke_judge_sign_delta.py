#!/usr/bin/env python3
"""Offline portable sign-plan import, paired review, reporting and resume."""

import json
import tempfile
from pathlib import Path

from tests.test_judge_sign_delta import smoke

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="sign-delta-smoke-") as directory:
        print(json.dumps(smoke(Path(directory)), indent=2))
