"""Old ACES Python must stop before parsing or importing recovery helpers."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import paired_fit_quota_recovery as entrypoint


def test_entrypoint_parses_with_python36_grammar():
    ast.parse(Path(entrypoint.__file__).read_text(), feature_version=(3, 6))


@pytest.mark.parametrize("version", [(3, 6, 8), (3, 8, 20)])
def test_old_python_stops_before_helper_imports(monkeypatch, version):
    monkeypatch.setattr(sys, "version_info", version)
    monkeypatch.setattr(sys, "argv", ["recovery.pyz", "cache", "--apply"])
    monkeypatch.setitem(sys.modules, "archive_finished_component_caches", None)
    with pytest.raises(SystemExit, match="No files were changed"):
        entrypoint.main()
    assert sys.argv == ["recovery.pyz", "cache", "--apply"]


@pytest.mark.parametrize(
    "command,module",
    [("cache", "archive_finished_component_caches"),
     ("install", "install_verified_paired_bundle")],
)
def test_supported_python_dispatches_without_changing_helper_arguments(
    monkeypatch, command, module,
):
    calls = []
    monkeypatch.setattr(sys, "version_info", (3, 9, 0))
    monkeypatch.setattr(sys, "argv", ["recovery.pyz", command, "--help"])
    monkeypatch.setitem(
        sys.modules, module, SimpleNamespace(main=lambda: calls.append(1))
    )
    entrypoint.main()
    assert calls == [1]
    assert sys.argv == ["recovery.pyz", "--help"]
