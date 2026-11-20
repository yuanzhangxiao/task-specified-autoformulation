"""Exercise the generic offline CLI on a synthetic rain-driven storage system."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search.topology_compatibility import (
    EvidenceContract,
    InitialRule,
    build_evidence,
)

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_topology_repairs.py"
)


def reservoir_input():
    """Two exact synthetic records of storage'=rain with identical initial storage."""
    rows = tuple(
        Trajectory(
            trajectory_id=name,
            time=np.array([0.0, 1.0, 2.0]),
            targets={"storage": np.array([2.0, 2.0 + rain, 2.0 + 2 * rain])},
            external_inputs={"rain": np.full(3, rain)},
            auxiliaries={},
            derivatives={},
            fixed_covariates={},
        )
        for name, rain in [("dry", 0.0), ("wet", 1.0)]
    )
    evidence = build_evidence(
        DatasetSplit(SplitName.TRAIN, rows, "synthetic linear reservoir"),
        ValidationContext(targets=("storage",), external_inputs=("rain",)),
        EvidenceContract(
            provenance="Noiseless synthetic test with constant rain on a common grid.",
            dependencies_exhaustive=True,
            shared_parameters=True,
            open_loop=True,
            reset_impulse_free=True,
            uniqueness_assumed=True,
            interpolation={"rain": "linear_complete_knots"},
            absolute_error_bounds={"storage": 0.0, "rain": 0.0},
            initial_rules={"storage": InitialRule(mode="observed_initial")},
        ),
    )
    return {
        "evidence": evidence.model_dump(mode="json"),
        "draft": {
            "variables": [
                {
                    "name": "storage",
                    "definition": "differential",
                    "scientific_role": "reservoir storage",
                }
            ],
            "equations": [
                {
                    "name": "storage",
                    "terms": [
                        {
                            "sources": ["storage"],
                            "scientific_role": "original drainage",
                            "outer_weight_sign": "negative",
                        }
                    ],
                }
            ],
        },
    }


def test_offline_cli_generates_direct_and_mediated_storage_repairs(tmp_path):
    source, out = tmp_path / "source.json", tmp_path / "report.json"
    source.write_text(json.dumps(reservoir_input()))
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input",
        str(source),
        "--output",
        str(out),
        "--target",
        "storage",
        "--left",
        "dry",
        "--right",
        "wet",
        "--time",
        "2",
        "--allowed-source",
        "rain",
        "--allowed-source",
        "storage",
    ]
    first = subprocess.run(cmd, check=True, text=True, capture_output=True)
    report = sealed_read(out)
    assert report["status_counts"] == {
        "distinguishes": 2,
        "still_forced_equal": 1,
        "invalid": 1,
    }
    direct = next(
        a
        for a in report["assessments"]
        if a["edit"]
        == {"kind": "add_dependency", "source": "rain", "target": "storage"}
    )
    assert direct["added_dependencies"] == [["rain", "storage"]]
    assert direct["remaining_conflicts"] == []
    # A retry reproduces the original sealed output.
    assert first.stdout == subprocess.check_output(cmd, text=True)
    # The generator validates the complete training digest before scoring.
    changed = json.loads(source.read_text())
    changed["evidence"]["trajectories"][0]["targets"]["storage"][0] += 1.0
    source.write_text(json.dumps(changed))
    failed = subprocess.run(cmd, capture_output=True, text=True)
    assert failed.returncode and "digest differs" in failed.stderr
    assert sealed_read(out) == report
