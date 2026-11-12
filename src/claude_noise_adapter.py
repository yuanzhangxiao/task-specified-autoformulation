"""The T2 noise handoff's adapter: package and forecast frozen baseline picks.

The handoff imports this module by name and binds its file hash; the method
code it calls is pinned by the configuration's implementation revision.
"""

from pathlib import Path

from autoformalism.rebuttal import noise_exchange


def fit(task: dict, cell_dir: Path, task_dir: Path, config: dict) -> dict:
    """Package one acquisition and seed's sealed pick; never search."""
    return noise_exchange.fit(task, Path(cell_dir), Path(task_dir), config)


def predict(model_path: Path, inputs: dict, solver: str, config: dict) -> dict:
    """Both targets for one blind query, from the frozen model file."""
    return noise_exchange.predict(Path(model_path), inputs, solver, config)
