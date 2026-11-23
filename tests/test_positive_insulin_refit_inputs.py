"""The matched-fit intake binds both public rows and original numerical semantics."""

import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from scripts import positive_insulin_refit_inputs as data
from tests.test_validate_positive_insulin_transfer import toy


def test_archive_checks_export_against_pinned_plan_and_rejects_extra_members(
    tmp_path, monkeypatch
):
    def record(d):
        return json.dumps(d | {"artifact_sha256": data.content_hash(d)})

    path = tmp_path / "bundle.zip"
    plan = {
        "models": {tag: {"name": tag} for tag in data.TAGS},
        "parent_ranking": "parent",
        "selected_positive_training_fit": "positive",
        "script": data.file_hash(Path(data.original.__file__)),
        "export_code": data.file_hash(Path(data.original.portable.__file__)),
    }
    digest = data.content_hash(plan)
    monkeypatch.setattr(data, "PLAN_DIGEST", digest)

    def write_bundle(models):
        with ZipFile(path, "w") as z:
            z.writestr("ORION.md", "")
            z.writestr("plan.json", record(plan))
            for tag, model in models.items():
                z.writestr(
                    f"models/{tag}.json",
                    record(
                        {
                            "source_plan": digest,
                            "training_archived_parent": "parent",
                            "selected_positive_full_training_fit": "positive"
                            if "positive" in tag
                            else None,
                            "full_training_model": model,
                        }
                    ),
                )

    write_bundle(plan["models"])
    assert data.load_models(path) == plan["models"]
    write_bundle(plan["models"] | {data.TAGS[1]: plan["models"][data.TAGS[0]]})
    with pytest.raises(ValueError, match="export differs"):
        data.load_models(path)
    with ZipFile(path, "a") as z:
        z.writestr("unexpected.json", "{}")
    with pytest.raises(ValueError, match="member"):
        data.load_models(path)


def test_split_gate_precedes_opening_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data, "sealed_read", lambda _: pytest.fail("opened wrong split")
    )
    with pytest.raises(ValueError, match=r"train\.json"):
        data.load_training(tmp_path / "validation.json")
    monkeypatch.setattr(
        data.original, "read_validation", lambda *_: pytest.fail("opened development")
    )
    with pytest.raises(ValueError, match="four frozen"):
        data.load_validation(tmp_path / "validation.json", {})


def test_original_forecast_with_updated_weights_and_no_later_observed_targets():
    row, m = toy("train_000")
    raw = vars(row)
    old = deepcopy(m)
    p = m["formula"]["parameters"] | {"pos_Gt_1": 3.5, "pos_tau": 4.0}
    fitted = data.with_parameters(m, p)
    assert m == old
    assert fitted["scalar_model"]["weights"][1] == 3.5
    assert fitted["scalar_model"]["tau"] == 4.0
    result = data.replay(fitted, [raw])
    e = result["events"][0]
    assert e["status"] == "scored"
    assert e["levels"][-1]["max_scalar_difference"] < 1e-5
    assert result["summary"]["pooled_nmse"] == pytest.approx(e["pooled_nmse"])
    observed = {n: [v[0], 100.0, 150.0, 200.0] for n, v in row.targets.items()}
    changed = data.replay(fitted, [raw | {"targets": observed}])["events"][0]
    assert changed["predicted"] == e["predicted"]
    assert changed["pooled_nmse"] != e["pooled_nmse"]
    assert not np.allclose(
        data.replay(m, [raw])["events"][0]["predicted"]["I"], e["predicted"]["I"]
    )


@pytest.mark.parametrize(
    "key,value", [("pos_tau", 0.0), ("pos_tau", np.inf), ("pos_Gt_1", -0.1)]
)
def test_physical_bounds_and_parameter_names_enforced(key, value):
    _, m = toy()
    with pytest.raises(ValueError):
        data.with_parameters(m, m["formula"]["parameters"] | {key: value})
    with pytest.raises(ValueError, match="names"):
        data.with_parameters(m, m["formula"]["parameters"] | {"extra": 1.0})


def test_replay_reports_numerically_unresolved_event(monkeypatch):
    row, model = toy("train_000")
    monkeypatch.setattr(data.original, "forecast", lambda *args: (None, [2, 4]))
    result = data.replay(model, [vars(row)])
    assert result["summary"]["pooled_nmse"] is None
    assert result["events"][0]["status"] == "grid_unresolved"
