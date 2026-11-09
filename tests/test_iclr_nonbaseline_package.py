"""Check the public discovery boundary and source-file fingerprint verification."""

from hashlib import sha256

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts.package_iclr_nonbaseline import PUBLIC_FILES, public_files


def make_release(root, *, extra_file=False):
    cells = []
    keys = ["acquisition_00", "sensor_00", *(f"condition_{i}" for i in range(19))]
    for key in keys:
        folder = root / "public" / key
        folder.mkdir(parents=True)
        files = {}
        for name in PUBLIC_FILES:
            data = f"{key}/{name}".encode()
            (folder / name).write_bytes(data)
            files[name] = sha256(data).hexdigest()
        if extra_file:
            files["test.json"] = "unavailable to discovery"
        cells.append({"cell": key, "public_files": files, "generator": "private"})
    sealed_write(
        root / "manifest.json",
        {
            "cells": cells,
            "test_data_generated": False,
            "provider_calls": 0,
            "runs": list(range(126)),
        },
    )


def test_discovery_archive_uses_curated_roster_and_detects_tampering(tmp_path):
    make_release(tmp_path)
    (tmp_path / "public/acquisition_00/test.json").write_text("evaluation-only")
    archive, roster = public_files(tmp_path)
    assert len(archive) == 84
    assert len(roster["stress"]) == 19
    assert not any("test.json" in name for name in archive)
    assert all(
        set(fields) == PUBLIC_FILES for fields in roster["public_files"].values()
    )
    (tmp_path / "public/sensor_00/train.json").write_text("changed")
    with pytest.raises(ValueError, match="changed public data"):
        public_files(tmp_path)


def test_extra_files_in_public_declaration_are_rejected(tmp_path):
    make_release(tmp_path, extra_file=True)
    with pytest.raises(ValueError, match="public-only payload"):
        public_files(tmp_path)
