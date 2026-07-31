"""The manual smoke script is inert unless explicitly authorized."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_smoke_script_without_live_makes_no_call(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(repository / "scripts/smoke_llm.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--live" in completed.stdout
    assert not (tmp_path / "artifacts").exists()
