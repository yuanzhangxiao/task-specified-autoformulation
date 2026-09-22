"""The Full extension must reuse exact cases with faithful public-channel aliases."""

from copy import deepcopy

import pytest

from scripts import probe_t1_full_interventions as full


def diagnostic_row():
    return {
        "trajectory_id": "probe",
        "time": [0, 1],
        "targets": {"v01": [172, 175]},
        "auxiliaries": {"v02": [2, 3], "v03": [1, 1], "v04": [0, 0], "v05": [130, 156]},
        "external_inputs": {"u01": [0, 60]},
    }


def test_aliases_preserve_values_and_do_not_mutate_source():
    row = diagnostic_row()
    before = deepcopy(row)
    named = full.adapt_channels(row, full.CELLS[0])
    obfuscated = full.adapt_channels(row, full.CELLS[1])
    assert obfuscated == row == before
    for role, mapping in full.CHANNELS.items():
        for alias, name in mapping.items():
            assert named[role][name] == row[role][alias]
    named["targets"]["Gp"][0] = 999
    assert row == before


def test_unknown_cells_and_incomplete_auxiliaries_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        full.adapt_channels(diagnostic_row(), "different_cell")
    row = diagnostic_row()
    del row["auxiliaries"]["v05"]
    with pytest.raises(ValueError, match="channels"):
        full.adapt_channels(row, full.CELLS[0])


def test_import_retains_digest_and_refuses_changed_data(tmp_path):
    source, destination = tmp_path / "source.json", tmp_path / "destination.json"
    original = full.sealed_write(source, {"row": diagnostic_row()})
    assert full.copy_sealed(source, destination) == original["artifact_sha256"]
    assert full.sealed_read(destination) == original
    assert full.copy_sealed(source, destination) == original["artifact_sha256"]
    destination.write_text(destination.read_text().replace("172", "999"))
    with pytest.raises(ValueError, match="digest differs"):
        full.copy_sealed(source, destination)
