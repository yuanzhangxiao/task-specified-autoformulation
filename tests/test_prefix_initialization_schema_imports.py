"""Construction schemas and legacy entrypoints do not require optional CasADi."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy


@pytest.mark.parametrize(
    "entrypoint", ["legacy_runner", "dependency_runner", "schemas"]
)
def test_construction_imports_without_casadi(entrypoint):
    """Exercise fresh processes so an already-loaded backend cannot hide imports."""
    source = """
import importlib.abc
import sys

class NoCasadi(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'casadi' or fullname.startswith('casadi.'):
            raise ModuleNotFoundError('optional CasADi blocked by import smoke test')

sys.meta_path.insert(0, NoCasadi())
if sys.argv[1] == 'legacy_runner':
    import autoformalism.search.staged_function_runner
elif sys.argv[1] == 'dependency_runner':
    import autoformalism.search.construction_graph_runner
else:
    import autoformalism.schemas.prefix_initialization
"""
    completed = subprocess.run(
        [sys.executable, "-c", source, entrypoint],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_numerical_api_reexports_the_identical_schema_classes():
    """Historical imports and the lightweight boundary share one validation API."""
    pytest.importorskip("casadi")
    from autoformalism.fitting import prefix_initialization

    assert prefix_initialization.PrefixFitProblem is PrefixFitProblem
    assert prefix_initialization.PrefixPolicy is PrefixPolicy
    for schema in (PrefixFitProblem, PrefixPolicy):
        public = getattr(prefix_initialization, schema.__name__)
        assert public.model_json_schema() == schema.model_json_schema()
    policy = PrefixPolicy(duration=0.5, future_inputs=("u",))
    assert (
        prefix_initialization.PrefixPolicy.model_validate_json(policy.model_dump_json())
        == policy
    )
    assert json.loads(policy.model_dump_json()) == {
        "duration": 0.5,
        "future_inputs": ["u"],
    }
